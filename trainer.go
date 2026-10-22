package kafn

import (
	"math"
	"os"
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
	printer   *StatsPrinter
	verbosity int
}

// NewTrainer creates a new trainer
func NewTrainer(precision, verbosity, parallelism int) *OnlineTrainer {
	if precision == 0 {
		precision = 4
	}
	return &OnlineTrainer{
		printer:   NewStatsPrinter(precision),
		verbosity: verbosity,
	}
}

type internal struct {
	E      [][]Deepfloat64
	deltas []Deepfloat64
	loss   Loss
}

func newE(layers []*Layer) [][]Deepfloat64 {
	E := make([][]Deepfloat64, len(layers))
	for i, l := range layers {
		E[i] = make([]Deepfloat64, len(l.Neurons))
	}
	return E
}

func newTraining(layers []*Layer, lossType LossType) *internal {
	var numOut int
	if len(layers) > 0 {
		numOut = len(layers[len(layers)-1].Neurons)
	}
	return &internal{
		E:      newE(layers),
		deltas: make([]Deepfloat64, numOut),
		loss:   GetLoss(lossType),
	}
}

// Set new output prefix
func (t *OnlineTrainer) SetPrefix(prefix string) {
	t.printer.SetPrefix(prefix)
}

// Train trains n
func (t *OnlineTrainer) Train(n *Neural, examples, validation Examples, iterations uint32) {
	if len(examples) == 0 {
		return
	}

	t.internal = newTraining(n.Layers, n.Config.Loss)

	t.printer.Init(n)

	ts := time.Now()

	if n.Config.Smooth {
		n.Smooth()
	}

	for i := uint32(1); i <= iterations; i++ {
		n.Config.Epoch++
		isReporting := (t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0) || i == iterations
		if isReporting {
			for k := range t.E {
				for j := range t.E[k] {
					t.E[k][j] = DF(0)
				}
			}
		}
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], isReporting)
		}
		if isReporting {
			for e := range t.E {
				for j := range t.E[e] {
					t.E[e][j] = Div(t.E[e][j], DF(float64(len(examples))))
				}
			}
			if t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0 {
				n.TotalError = TotalError(t.E[len(n.Layers)-1])
				t.printer.PrintProgress(n, validation, time.Since(ts), i)
			}
		}
		t.epoch(n, uint32(n.Config.Epoch))
	}
	n.TotalError = TotalError(t.E[len(n.Layers)-1])
}

func (t *OnlineTrainer) learn(n *Neural, e Example, recordLoss bool) {
	n.Forward(e.Input)
	if t.calculateDeltas(n, e.Response, recordLoss) {
		t.update(n, t.deltas)
	}
}

func (t *OnlineTrainer) calculateDeltas(n *Neural, ideal []Deepfloat64, recordLoss bool) bool {
	needsUpdate := false
	outLayerIdx := len(n.Layers) - 1
	outputLayer := n.Layers[outLayerIdx]

	for i, neuron := range outputLayer.Neurons {
		// Calculate the error gradient (ideal - estimate)
		err := Sub(ideal[i], neuron.Sum)
		t.deltas[i] = err
		// Accumulate the squared error for reporting
		if recordLoss {
			t.E[outLayerIdx][i] = Add(t.E[outLayerIdx][i], t.loss.F(neuron.Sum, ideal[i]))
		}
		neuron.Ideal = ideal[i]
		if err != 0 {
			needsUpdate = true
		}
	}
	return needsUpdate
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

func (t *OnlineTrainer) update(neural *Neural, errors []Deepfloat64) {
	outLayerIdx := len(neural.Layers) - 1
	l := neural.Layers[outLayerIdx]

	for i, n := range l.Neurons {
		numIn := len(n.In)
		if numIn == 0 { // A neuron with no inputs cannot learn.
			continue
		}

		remainingError := errors[i]

		for idx, synapse := range n.In {
			remainingSynapses := numIn - idx
			gap := Div(remainingError, DF(float64(remainingSynapses)))

			currentOut := synapse.GetOut()
			newOut := Add(currentOut, gap)

			// If the gap is too small to change the float64 value, force a change
			// by finding the next representable floating-point number.
			if newOut == currentOut && gap != 0 {
				if gap > 0 {
					newOut = DF(math.Nextafter(Float64(currentOut), math.Inf(1)))
				} else {
					newOut = DF(math.Nextafter(Float64(currentOut), math.Inf(-1)))
				}
			}
			if newOut == 0 {
				newOut = DF(math.Nextafter(0.0, math.Inf(1)))
			}
			synapse.AddPoint(synapse.GetIn(), newOut, neural.Config.Epoch)

			actualAppliedGap := Sub(newOut, currentOut)
			remainingError = Sub(remainingError, actualAppliedGap)
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
