package kafn

import (
	"context"
	"os"
	"runtime"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/theothertomelliott/acyclic"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *Neural, examples, validation Examples, iterations int)
	SetPrefix(prefix string)
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	printer     *StatsPrinter
	verbosity   int
	parallelism int
	sem         *semaphore.Weighted
}

// NewTrainer creates a new trainer
func NewTrainer(precision, verbosity, parallelism int) *OnlineTrainer {
	if precision == 0 {
		precision = 4
	}
	if parallelism == 0 {
		parallelism = runtime.NumCPU()
	}
	return &OnlineTrainer{
		printer:     NewStatsPrinter(precision),
		verbosity:   verbosity,
		parallelism: parallelism,
		sem:         semaphore.NewWeighted(int64(parallelism)),
	}
}

type internal struct {
	E [][]Deepfloat64
}

func newE(layers []*Layer) [][]Deepfloat64 {
	E := make([][]Deepfloat64, len(layers))
	for i, l := range layers {
		E[i] = make([]Deepfloat64, len(l.Neurons))
	}
	return E
}

func newTraining(layers []*Layer) *internal {
	return &internal{
		E: newE(layers),
	}
}

// Set new output prtefix
func (t *OnlineTrainer) SetPrefix(prefix string) {
	t.printer.SetPrefix(prefix)
}

// Train trains n
func (t *OnlineTrainer) Train(n *Neural, examples, validation Examples, iterations uint32) {
	t.internal = newTraining(n.Layers)

	t.printer.Init(n)

	ts := time.Now()

	if n.Config.Smooth {
		n.Smooth()
	}

	for i := uint32(1); i <= iterations; i++ {
		examples.Shuffle()
		n.Config.Epoch++
		for i := range t.E {
			for j := range t.E[i] {
				t.E[i][j] = 0
			}
		}
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], uint32(n.Config.Epoch))
		}
		for e := range t.E {
			for j := range t.E[e] {
				t.E[e][j] /= Deepfloat64(len(examples))
			}
		}
		if t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0 {
			n.TotalError = TotalError(t.E[len(n.Layers)-1])
			t.printer.PrintProgress(n, validation, time.Since(ts), i)
		}
		t.epoch(n, uint32(n.Config.Epoch))
	}
	n.TotalError = TotalError(t.E[len(n.Layers)-1])
}

func (t *OnlineTrainer) learn(n *Neural, e Example, it uint32) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n, it)
}

func (t *OnlineTrainer) calculateDeltas(n *Neural, ideal []Deepfloat64) {
	loss := GetLoss(n.Config.Loss)
	var wg sync.WaitGroup
	ctx := context.Background()
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.sem.Acquire(ctx, 1)
		wg.Add(1)
		go func(wg *sync.WaitGroup, neuron *Neuron, i int) {
			defer t.sem.Release(1)
			t.E[len(n.Layers)-1][i] += loss.F(neuron.Sum, ideal[i])
			neuron.Ideal = ideal[i]

			wg.Done()
		}(&wg, neuron, i)
	}
	wg.Wait()

	for i := len(n.Layers) - 2; i >= 1; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			t.sem.Acquire(ctx, 1)
			wg.Add(1)
			go func(wg *sync.WaitGroup, neuron *Neuron, i, j int) {
				defer t.sem.Release(1)
				var n_ideal Deepfloat64
				for _, s := range neuron.Out {

					var gap Deepfloat64
					numIn := len(s.GetUp().In)
					if numIn > 0 {
						gap = (s.GetUp().Ideal - s.GetUp().Sum) / Deepfloat64(numIn)
					}
					if s.Len() > 1 && s.GetWeight(1) != 0 {
						n_ideal += s.GetIn() + (gap-s.GetWeight(0))/s.GetWeight(1)
					}

				}
				n_ideal = n_ideal / Deepfloat64(len(neuron.Out))
				t.E[i][j] += loss.F(neuron.Sum, n_ideal)
				neuron.Ideal = n_ideal

				wg.Done()
			}(&wg, neuron, i, j)
		}
		wg.Wait()
	}
}

// Set epoch for Tabulated Activations
func (t *OnlineTrainer) epoch(neural *Neural, epoch uint32) {
	for i := len(neural.Layers) - 1; i >= 1; i-- {
		l := neural.Layers[i]
		for _, n := range l.Neurons {
			if l.S == SynapseTypeTabulated {
				for _, s := range n.In {
					s.Epoch(epoch)
				}
			}
		}
	}

}

// Update from bootom up
func (t *OnlineTrainer) update(neural *Neural, it uint32) {
	var wg sync.WaitGroup
	for i, l := range neural.Layers {
		if i == 0 {
			continue
		}
		for j, n := range l.Neurons {
			t.sem.Acquire(context.Background(), 1)
			wg.Add(1)
			go func(wg *sync.WaitGroup, n *Neuron, i, j int, l *Layer) {
				defer t.sem.Release(1)

				gap := (n.Ideal - n.Sum) / Deepfloat64(len(n.In)+1)

				for _, synapse := range n.In {
					switch l.S {
					case SynapseTypeTabulated:
						synapse.AddPoint(synapse.GetIn(), synapse.GetOut()+gap, neural.Config.Epoch)
					case SynapseTypeAnalytic:
					}
				}

				wg.Done()
			}(&wg, n, i, j, l)
		}
		wg.Wait()
	}
}

// Save() saves internal of the trainer in readable JSON into file specified
func (t *OnlineTrainer) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	acyclic.Fprint(f, t.internal)
	return nil
}
