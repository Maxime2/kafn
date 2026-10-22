package kafn

import (
	"fmt"
	"log"
	"strings"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

// Smallest number
const Eps = 1e-16

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
	// Sum of weights
	sum_of_weights Deepfloat64
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
	layers := []*Layer{
		NewLayer(0, 2*c.Inputs+1, SynapseTypeAnalytic),
		NewLayer(1, c.Outputs, SynapseTypeTabulated),
	}

	layers[0].CreateInputSynapses(c)
	layers[0].Connect(layers[1], c)

	return layers
}

func (n *Neural) fire() {
	for _, neuron := range n.Layers[0].Neurons {
		neuron.calculateNeuronOutput()
	}
	if len(n.Layers) > 1 {
		l0 := n.Layers[0]
		for _, neuron := range n.Layers[1].Neurons {
			for i, s := range neuron.TabulatedIn {
				s.Fire(l0.Neurons[i].Sum)
			}
			neuron.calculateNeuronOutput()
		}
	}
}

func (n *Neural) fireT(trapolation tabulatedfunction.Trapolation) {
	for _, neuron := range n.Layers[0].Neurons {
		neuron.calculateNeuronOutput()
	}
	if len(n.Layers) > 1 {
		l0 := n.Layers[0]
		for _, neuron := range n.Layers[1].Neurons {
			for i, s := range neuron.TabulatedIn {
				s.FireT(l0.Neurons[i].Sum, trapolation)
			}
			neuron.calculateNeuronOutput()
		}
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []Deepfloat64) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, neuron := range n.Layers[0].Neurons {
		for i := range input {
			neuron.AnalyticIn[i].Fire(input[i])
		}
	}

	n.fire()
	return nil
}

// ForwardT computes a forward pass with trapolation
func (n *Neural) ForwardT(input []Deepfloat64, trapolation tabulatedfunction.Trapolation) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, neuron := range n.Layers[0].Neurons {
		for i := range input {
			neuron.AnalyticIn[i].FireT(input[i], trapolation)
		}
	}

	n.fireT(trapolation)
	return nil
}

// PredictInto computes a forward pass and copies the prediction into out.
func (n *Neural) PredictInto(input []Deepfloat64, out []Deepfloat64) error {
	if err := n.Forward(input); err != nil {
		return err
	}
	outLayer := n.Layers[len(n.Layers)-1]
	if len(out) != len(outLayer.Neurons) {
		return fmt.Errorf("invalid output slice dimension - expected: %d got: %d", len(outLayer.Neurons), len(out))
	}
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Sum
	}
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []Deepfloat64) []Deepfloat64 {
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	if err := n.PredictInto(input, out); err != nil {
		panic(fmt.Sprintf("prediction failed: %v", err))
	}
	return out
}

// TrapolateInto computes a forward pass with trapolation and copies the prediction into out.
func (n *Neural) TrapolateInto(input []Deepfloat64, out []Deepfloat64, trapolation tabulatedfunction.Trapolation) error {
	if err := n.ForwardT(input, trapolation); err != nil {
		return err
	}
	outLayer := n.Layers[len(n.Layers)-1]
	if len(out) != len(outLayer.Neurons) {
		return fmt.Errorf("invalid output slice dimension - expected: %d got: %d", len(outLayer.Neurons), len(out))
	}
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Sum
	}
	return nil
}

// Trapolate computes a forward pass with trapolation and returns a prediction
func (n *Neural) Trapolate(input []Deepfloat64, trapolation tabulatedfunction.Trapolation) []Deepfloat64 {
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	if err := n.TrapolateInto(input, out, trapolation); err != nil {
		panic(fmt.Sprintf("prediction failed: %v", err))
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, neuron := range l.Neurons {
			for _, synapse := range neuron.AnalyticIn {
				num += synapse.Len()
			}
			for _, synapse := range neuron.TabulatedIn {
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
			for x, in := range neuron.TabulatedIn {
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
			for _, in := range neuron.TabulatedIn {
				in.Smooth()
			}
		}
	}
}

func (n *Neural) Polinate() {
	if len(n.Layers) < 2 {
		return
	}
	l := n.Layers[1] // Target the top layer explicitly
	for y, neuron := range l.Neurons {
		for x, in := range neuron.TabulatedIn {
			for t, p := range l.Neurons {
				if t <= y {
					continue
				}
				in.Polinate(p.TabulatedIn[x])
			}
		}
	}
}

func (n *Neural) Expand(steps int) {
	if len(n.Layers) < 2 {
		return
	}
	l := n.Layers[1] // Target the top layer explicitly
	for _, neuron := range l.Neurons {
		for _, s := range neuron.TabulatedIn {
			s.Expand(steps)
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
