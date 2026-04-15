package kafn

import (
	"fmt"
	"log"
	"strings"
	"sync"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

// Smallest number
const Eps = 1e-16
const Leps = 1e-20

// Minimal number of iterations
const MinIterations = 5

// Neural is a neural network
type Neural struct {
	Layers     []*Layer
	Config     *Config
	TotalError Deepfloat64
}

// Config defines the network topology, activations, losses etc
type Config struct {
	// Number of inputs
	Inputs int
	// Number of outputs
	Outputs int
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Error/Loss precision
	LossPrecision int
	// Specifies basis size
	Degree int
	// Specify Synap Tags for the input layer
	InputTags []string
	// Number of training iterations
	Epoch uint32
	// If Smooth() need to be executed before each training iteration
	Smooth bool
	// TrapolationLinear by default
	Trapolation tabulatedfunction.Trapolation
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.InputTags == nil {
		c.InputTags = make([]string, c.Inputs)
		for i := range c.InputTags {
			c.InputTags[i] = fmt.Sprintf("In:%d", i)
		}
	}
	if c.Loss == LossNone {
		c.Loss = LossMeanSquared
	}
	if c.LossPrecision == 0 {
		c.LossPrecision = 4
	}

	if c.Degree == 0 {
		c.Degree = 1
	}

	if c.Degree != 1 {
		log.Fatal("Degree must be 1; or unspecified!")
	}

	layers := initializeLayers(c)

	return &Neural{
		Layers: layers,
		Config: c,
	}
}

func initializeLayers(c *Config) []*Layer {
	var layout []int
	var synapse []SynapseType

	layout = []int{2*c.Inputs + 1, c.Outputs}
	synapse = []SynapseType{SynapseTypeAnalytic, SynapseTypeTabulated}

	layers := make([]*Layer, len(layout))

	for i := range layers {
		layers[i] = NewLayer(i, layout[i], synapse[i])
	}

	layers[0].CreateInputSynapses(c)

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c)
	}

	return layers
}

func (n *Neural) fire() {
	for _, l := range n.Layers {
		l.Fire()
	}
}

func (n *Neural) fireT(trapolation tabulatedfunction.Trapolation) {
	for _, l := range n.Layers {
		l.FireT(trapolation)
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []Deepfloat64) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	var wg sync.WaitGroup
	for _, neuron := range n.Layers[0].Neurons {
		wg.Add(1)
		go func(neuron *Neuron) {
			defer wg.Done()
			for i := range input {
				neuron.In[i].Fire(input[i])
			}
		}(neuron)
	}
	wg.Wait()

	n.fire()
	return nil
}

// ForwardT computes a forward pass with trapolation
func (n *Neural) ForwardT(input []Deepfloat64, trapolation tabulatedfunction.Trapolation) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	var wg sync.WaitGroup
	for _, neuron := range n.Layers[0].Neurons {
		wg.Add(1)
		go func(neuron *Neuron) {
			defer wg.Done()
			for i := range input {
				neuron.In[i].FireT(input[i], trapolation)
			}
		}(neuron)
	}
	wg.Wait()

	n.fireT(trapolation)
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []Deepfloat64) []Deepfloat64 {
	err := n.Forward(input)
	if err != nil {
		// A panic is appropriate here because incorrect input dimensions are a programmer error.
		panic(fmt.Sprintf("prediction failed: %v", err))
	}
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Sum
	}
	return out
}

// Trapolate computes a forward pass with trapolation and returns a prediction
func (n *Neural) Trapolate(input []Deepfloat64, trapolation tabulatedfunction.Trapolation) []Deepfloat64 {
	err := n.ForwardT(input, trapolation)
	if err != nil {
		// A panic is appropriate here because incorrect input dimensions are a programmer error.
		panic(fmt.Sprintf("prediction failed: %v", err))
	}
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Sum
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, neuron := range l.Neurons {
			for _, synapse := range neuron.In {
				num += synapse.Len()
			}
		}
	}
	return
}

func (n *Neural) String() string {
	var sb strings.Builder
	for i, l := range n.Layers {
		if i > 0 {
			sb.WriteByte('\n')
		}
		// Fprint will use the String() method of the Layer if it exists.
		fmt.Fprint(&sb, l)
	}
	return sb.String()
}

func TotalError(E []Deepfloat64) Deepfloat64 {
	r := DF(0)
	for _, x := range E {
		r = Add(r, x)
	}
	return r
}

func (n *Neural) DrawPS(path_prefix string) {
	for _, l := range n.Layers {
		for y, neuron := range l.Neurons {
			for x, in := range neuron.In {
				path := fmt.Sprintf("%sL-%v-N-%v-In-%v.ps", path_prefix, l.Number, y, x)
				in.DrawPS(path)
			}
		}
	}
}

func (n *Neural) Smooth() {
	bottom := 1

	for a, l := range n.Layers {
		if a < bottom {
			continue
		}

		for _, neuron := range l.Neurons {
			for _, in := range neuron.In {
				in.Smooth()
			}
		}
	}
}

func (n *Neural) Polinate() {
	bottom := 1

	for a, l := range n.Layers {
		if a < bottom {
			continue
		}

		for y, neuron := range l.Neurons {
			for x, in := range neuron.In {
				for t, p := range l.Neurons {
					if t <= y {
						continue
					}
					m := p.In[x]
					in.Polinate(m)
				}
			}
		}
	}
}

func (n *Neural) Check() string {
	var report string = ""

	Layer := n.Layers[0]
	for y, neuron := range Layer.Neurons {
		for t, p := range Layer.Neurons {
			if t <= y {
				continue
			}
			if neuron.MinSum > p.MinSum && neuron.MinSum < p.MaxSum {
				report += fmt.Sprintf("Check %d vs %d: [%v:%v] vs [%v:%v]\n",
					y, t, neuron.MinSum, neuron.MaxSum, p.MinSum, p.MaxSum)
			}
		}

	}

	return report
}
