package kafn

import (
	"fmt"
	"math"
)

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Number  int
	S       SynapseType
	Neurons []*Neuron
}

// NewLayer creates a new layer with n nodes
func NewLayer(l, n int, synapse SynapseType) *Layer {
	//func NewLayer(c *Config, l int) *Layer {
	//	n := c.Layout[l]
	//	activation := c.Activation[l]
	//	synapse := c.Synapse[l]

	neurons := make([]*Neuron, n)
	storage := make([]Neuron, n)

	for i := 0; i < n; i++ {
		storage[i].MinSum = math.MaxFloat64
		storage[i].MaxSum = -math.MaxFloat64
		neurons[i] = &storage[i]
	}
	return &Layer{
		Number:  l,
		Neurons: neurons,
		S:       synapse,
	}
}

// CreateInputSynapses create input synapses for the bottom layer
func (l *Layer) CreateInputSynapses(c *Config) {
	if c.Inputs <= 0 {
		return
	}
	wA := DF(0)
	c.sum_of_weights = wA
	for _, neuron := range l.Neurons {
		synStorage := make([]SynapseAnalytic, c.Inputs)
		neuron.AnalyticIn = make([]*SynapseAnalytic, c.Inputs)
		for i := range neuron.AnalyticIn {
			// Nested logarithms (iterated logarithm) provide even slower growth.
			A := 0.5 * math.Log(3.0+math.Log(3.0+math.Log(3.0+float64(i+1)/float64(c.Inputs)))) / float64(2*c.Inputs+1)
			tag := ""
			if i < len(c.InputTags) {
				tag = c.InputTags[i]
			} else {
				tag = fmt.Sprintf("In:%d", i)
			}
			weights := make([]float64, c.Degree+1)
			weights[0] = wA
			weights[1] = DF(A)
			synStorage[i] = SynapseAnalytic{
				Weights: weights,
				Up:      neuron,
				Tag:     tag,
			}
			neuron.AnalyticIn[i] = &synStorage[i]
			c.sum_of_weights += wA
			c.sum_of_weights += DF(A)
			wA += DF(A + Eps)
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
// func (l *Layer) Connect(next *Layer, degree int, weight WeightType) {
func (l *Layer) Connect(next *Layer, c *Config) {
	l.connectInternal(next, c, true)
}

func (l *Layer) connectInternal(next *Layer, c *Config, initPoints bool) {
	numIn := len(l.Neurons)
	for _, neuron := range next.Neurons {
		synStorage := make([]SynapseTabulated, numIn)
		neuron.TabulatedIn = make([]*SynapseTabulated, numIn)
		for i := range l.Neurons {
			tag := fmt.Sprintf("L:%d N:%d", l.Number, i)
			initSynapseTabulated(&synStorage[i], c, neuron, tag, initPoints)
			if initPoints {
				synStorage[i].AddPoint(0.5, 0.5, 0)
			}
			neuron.TabulatedIn[i] = &synStorage[i]
		}
	}
}

func (l *Layer) String() string {
	// The original implementation `return fmt.Sprintf("%+v", l)` causes infinite recursion.
	return fmt.Sprintf("Layer(Number: %d, SynapseType: %v, Neurons: %d)", l.Number, l.S, len(l.Neurons))
}
