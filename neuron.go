package kafn

import (
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

// Neuron is a neural network node
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

func (n *Neuron) calculateAndFire(refireSynapses bool) {
	n.Sum = DF(0)
	for _, s := range n.In {
		if refireSynapses {
			s.Refire()
		}
		preliminarySum := Add(n.Sum, s.GetOut())
		if !math.IsNaN(Float64(preliminarySum)) {
			n.Sum = preliminarySum
		}
	}

	if n.Sum.Cmp(n.MinSum) < 0 {
		n.MinSum = n.Sum
	}
	if n.Sum.Cmp(n.MaxSum) > 0 {
		n.MaxSum = n.Sum
	}

	nVal := n.Sum
	for _, s := range n.Out {
		s.Fire(nVal)
	}
}

func (n *Neuron) fire() {
	n.calculateAndFire(false)
}

func (n *Neuron) refire() {
	n.calculateAndFire(true)
}

func (n *Neuron) fireT(trapolation tabulatedfunction.Trapolation) {
	n.Sum = DF(0)
	for _, s := range n.In {
		preliminarySum := Add(n.Sum, s.GetOut())
		if !math.IsNaN(Float64(preliminarySum)) {
			n.Sum = preliminarySum
		}
	}

	if n.Sum.Cmp(n.MinSum) < 0 {
		n.MinSum = n.Sum
	}
	if n.Sum.Cmp(n.MaxSum) > 0 {
		n.MaxSum = n.Sum
	}

	nVal := n.Sum
	for _, s := range n.Out {
		s.FireT(nVal, trapolation)
	}
}
