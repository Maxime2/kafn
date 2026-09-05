package kafn

import (
	"fmt"
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
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

	for i := 0; i < n; i++ {
		neurons[i] = NewNeuron()
	}
	return &Layer{
		Number:  l,
		Neurons: neurons,
		S:       synapse,
	}
}

func (l *Layer) Fire() {
	for _, neuron := range l.Neurons {
		neuron.fire()
	}
}

func (l *Layer) FireT(trapolation tabulatedfunction.Trapolation) {
	for _, neuron := range l.Neurons {
		neuron.fireT(trapolation)
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
		neuron.In = make([]Synapse, c.Inputs)
		for i := range neuron.In {
			// Nested logarithms (iterated logarithm) provide even slower growth.
			A := 0.5 * math.Log(3.0+math.Log(3.0+math.Log(3.0+float64(i+1)/float64(c.Inputs)))) / float64(2*c.Inputs+1)
			tag := fmt.Sprintf("In:%d", i)
			if i < len(c.InputTags) {
				tag = c.InputTags[i]
			}
			neuron.In[i] = NewSynapseAnalytic(neuron, c.Degree, []Deepfloat64{wA, DF(A)}, tag)
			c.sum_of_weights = Add(c.sum_of_weights, wA)
			c.sum_of_weights = Add(c.sum_of_weights, DF(A))
			wA = Add(wA, DF(A+Eps))
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
// func (l *Layer) Connect(next *Layer, degree int, weight WeightType) {
func (l *Layer) Connect(next *Layer, c *Config) {
	for _, neuron := range next.Neurons {
		for i := range l.Neurons {
			syn := NewSynapseTabulated(c, neuron, fmt.Sprintf("L:%d N:%d", l.Number, i))
			syn.AddPoint(0.5, 0.5, 0)
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			neuron.In = append(neuron.In, syn)
		}
	}
}

func (l *Layer) NumIns() (num int) {
	for _, neuron := range l.Neurons {
		num += len(neuron.In)
	}
	return
}

func (l *Layer) String() string {
	// The original implementation `return fmt.Sprintf("%+v", l)` causes infinite recursion.
	return fmt.Sprintf("Layer(Number: %d, SynapseType: %v, Neurons: %d)", l.Number, l.S, len(l.Neurons))
}
