package kafn

import (
	"log"
	"math"
)

// Neuron is a neural network node
// Note: Neuron stores topology and boundary tracking (MinSum/MaxSum).
// For thread-safe concurrent evaluation and parallel training, use Context.
type Neuron struct {
	AnalyticIn     []*SynapseAnalytic
	TabulatedIn    []*SynapseTabulated
	Ideal          float64
	Sum            float64
	MinSum, MaxSum float64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron() *Neuron {
	return &Neuron{
		MinSum: DF(math.MaxFloat64),  // Correct for finding a minimum
		MaxSum: DF(-math.MaxFloat64), // Correct for finding a maximum
	}
}

func (n *Neuron) setSum(sum float64) float64 {
	n.Sum = sum
	val := n.Sum
	if math.IsNaN(val) || math.IsInf(val, 0) {
		log.Printf("WARN: Invalid number (NaN/Inf) detected during neuron sum calculation. Resetting sum to 0.")
		n.Sum = DF(0)
	}
	if n.Sum < n.MinSum {
		n.MinSum = n.Sum
	}
	if n.Sum > n.MaxSum {
		n.MaxSum = n.Sum
	}
	return n.Sum
}

func (n *Neuron) calculateNeuronOutput() float64 {
	var sum float64
	for _, s := range n.AnalyticIn {
		sum += s.Out
	}
	for _, s := range n.TabulatedIn {
		sum += s.Out
	}
	return n.setSum(sum)
}
