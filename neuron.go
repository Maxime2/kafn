package kafn

import (
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

// Neuron is a neural network node
//
// Note: Neuron is stateful and NOT thread-safe. Concurrent calls to
// Forward or Predict on the parent Neural network will cause data races
// and corrupted activation states because transient values are stored directly in the fields.
type Neuron struct {
	In             []Synapse
	Out            []Synapse
	Ideal          Deepfloat64
	Sum            Deepfloat64
	MinSum, MaxSum Deepfloat64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron() *Neuron {
	return &Neuron{
		MinSum: DF(math.MaxFloat64),  // Correct for finding a minimum
		MaxSum: DF(-math.MaxFloat64), // Correct for finding a maximum
	}
}

func (n *Neuron) calculateNeuronOutput() Deepfloat64 {
	n.Sum = DF(0)
	for _, s := range n.In {
		preliminarySum := Add(n.Sum, s.GetOut())
		val := Float64(preliminarySum)
		if !math.IsNaN(val) && !math.IsInf(val, 0) {
			n.Sum = preliminarySum
		}
	}
	if n.Sum < n.MinSum {
		n.MinSum = n.Sum
	}
	if n.Sum > n.MaxSum {
		n.MaxSum = n.Sum
	}
	return n.Sum
}

func (n *Neuron) fire() {
	nVal := n.calculateNeuronOutput()

	for _, s := range n.Out {
		s.Fire(nVal)
	}
}

func (n *Neuron) fireT(trapolation tabulatedfunction.Trapolation) {
	nVal := n.calculateNeuronOutput()

	for _, s := range n.Out {
		s.FireT(nVal, trapolation)
	}
}
