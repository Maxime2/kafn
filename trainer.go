package kafn

import (
	"os"
	"runtime"
	"time"

	"github.com/theothertomelliott/acyclic"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *Neural, examples, validation Examples, iterations uint32)
	SetPrefix(prefix string)
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	printer     *StatsPrinter
	verbosity   int
	parallelism int
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

// Set new output prefix
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
		for k := range t.E {
			for j := range t.E[k] {
				t.E[k][j] = DF(0)
			}
		}
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j])
		}
		for e := range t.E {
			for j := range t.E[e] {
				t.E[e][j] = Div(t.E[e][j], DF(float64(len(examples))))
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

func (t *OnlineTrainer) learn(n *Neural, e Example) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n)
}

func (t *OnlineTrainer) calculateDeltas(n *Neural, ideal []Deepfloat64) {
	loss := GetLoss(n.Config.Loss)
	for i, neuron := range n.Layers[1].Neurons {
		// Spawning goroutines per output neuron for each example adds significant overhead.
		// Sequential processing is more efficient for typical output layer sizes.
		t.E[1][i] = Add(t.E[1][i], loss.F(neuron.Sum, ideal[i]))
		neuron.Ideal = ideal[i]
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

// Update from bottom up
func (t *OnlineTrainer) update(neural *Neural) {
	l := neural.Layers[1]

	for _, n := range l.Neurons {
		numIn := len(n.In)
		if numIn == 0 {
			numIn = 1
		}
		gap := Div(n.Ideal, DF(float64(numIn)))

		for _, synapse := range n.In {
			synapse.AddPoint(synapse.GetIn(), gap, neural.Config.Epoch)
		}
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
