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
	wA := DF(0)
	for _, neuron := range l.Neurons {
		neuron.In = make([]Synapse, c.Inputs)
		for i := range neuron.In {
			// Nested logarithms (iterated logarithm) provide even slower growth.
			A := 0.5 * math.Log(3.0+math.Log(3.0+math.Log(3.0+float64(i+1)/float64(c.Inputs)))) / float64(2*c.Inputs+1)
			neuron.In[i] = NewSynapseAnalytic(neuron, c.Degree, []Deepfloat64{wA, DF(A)}, c.InputTags[i])
			neuron.In[i].SetWeight(0, wA)
			neuron.In[i].SetWeight(1, DF(A))
			wA = Add(wA, DF(A+Eps))
		}
	}
	for _, neuron := range l.Neurons {
		for _, s := range neuron.In {
			s.SetWeight(0, s.GetWeight(0)/wA)
			s.SetWeight(1, s.GetWeight(1)/wA)
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
// func (l *Layer) Connect(next *Layer, degree int, weight WeightType) {
func (l *Layer) Connect(next *Layer, c *Config) {
	for j, neuron := range next.Neurons {
		for i := range l.Neurons {
			syn := NewSynapseTabulated(c, neuron, fmt.Sprintf("L:%d N:%d", l.Number, i))
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			next.Neurons[j].In = append(next.Neurons[j].In, syn)
		}
	}
}

func (l Layer) NumIns() (num int) {
	for _, neuron := range l.Neurons {
		num += len(neuron.In)
	}
	return
}

func (l Layer) String() string {
	// The original implementation `return fmt.Sprintf("%+v", l)` causes infinite recursion.
	return fmt.Sprintf("Layer(Number: %d, SynapseType: %v, Neurons: %d)", l.Number, l.S, len(l.Neurons))
}
