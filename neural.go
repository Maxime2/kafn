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

// Neural is a neural network
type Neural struct {
	Layers     []*Layer
	Config     *Config
	TotalError Deepfloat64
	initOnce   sync.Once
	ctxPool    sync.Pool
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

// Context holds transient execution state for forward passes and predictions.
// Using a Context decouples transient activations from Neural/Neuron/Synapse structs,
// enabling thread-safe concurrent evaluation and parallel training.
type Context struct {
	LayerSums   [][]Deepfloat64 // [layer][neuron]
	TabulatedIn [][]Deepfloat64 // [outputNeuron][inputNeuron] Out
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

	n := &Neural{
		Layers: layers,
		Config: c,
	}
	n.initPool()
	return n
}

func (n *Neural) initPool() {
	n.initOnce.Do(func() {
		n.ctxPool = sync.Pool{
			New: func() any {
				return n.NewContext()
			},
		}
	})
}

func (n *Neural) getContext() *Context {
	n.initPool()
	return n.ctxPool.Get().(*Context)
}

// NewContext creates an isolated execution buffer sized for the network topology.
func (n *Neural) NewContext() *Context {
	ctx := &Context{
		LayerSums:   make([][]Deepfloat64, len(n.Layers)),
		TabulatedIn: make([][]Deepfloat64, 0),
	}
	for i, l := range n.Layers {
		ctx.LayerSums[i] = make([]Deepfloat64, len(l.Neurons))
	}
	if len(n.Layers) > 1 {
		l1 := n.Layers[1]
		ctx.TabulatedIn = make([][]Deepfloat64, len(l1.Neurons))
		l0Len := len(n.Layers[0].Neurons)
		for i := range l1.Neurons {
			ctx.TabulatedIn[i] = make([]Deepfloat64, l0Len)
		}
	}
	return ctx
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
			var sum Deepfloat64
			for i, s := range neuron.TabulatedIn {
				s.Fire(l0.Neurons[i].Sum)
				sum += s.Out
			}
			neuron.setSum(sum)
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
			var sum Deepfloat64
			for i, s := range neuron.TabulatedIn {
				s.FireT(l0.Neurons[i].Sum, trapolation)
				sum += s.Out
			}
			neuron.setSum(sum)
		}
	}
}

// ForwardCtx computes a forward pass using an explicit execution context without mutating node state.
func (n *Neural) ForwardCtx(input []Deepfloat64, ctx *Context) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	l0 := n.Layers[0]
	l0Sums := ctx.LayerSums[0]
	for neuronIdx, neuron := range l0.Neurons {
		var sum Deepfloat64
		for i, val := range input {
			s := neuron.AnalyticIn[i]
			nw := len(s.Weights)
			if nw == 2 {
				sum += s.Weights[0] + s.Weights[1]*val
			} else if nw > 0 {
				out := s.Weights[nw-1]
				for k := nw - 2; k >= 0; k-- {
					out = out*val + s.Weights[k]
				}
				sum += out
			}
		}
		l0Sums[neuronIdx] = sum
	}

	if len(n.Layers) > 1 {
		l1 := n.Layers[1]
		l1Sums := ctx.LayerSums[1]
		for neuronIdx, neuron := range l1.Neurons {
			var sum Deepfloat64
			tabOut := ctx.TabulatedIn[neuronIdx]
			for i, s := range neuron.TabulatedIn {
				out := Deepfloat64(s.direct.F(float64(l0Sums[i])))
				tabOut[i] = out
				sum += out
			}
			l1Sums[neuronIdx] = sum
		}
	}
	return nil
}

// ForwardTCtx computes a forward pass with trapolation using an explicit execution context.
func (n *Neural) ForwardTCtx(input []Deepfloat64, trapolation tabulatedfunction.Trapolation, ctx *Context) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	l0 := n.Layers[0]
	l0Sums := ctx.LayerSums[0]
	for neuronIdx, neuron := range l0.Neurons {
		var sum Deepfloat64
		for i, val := range input {
			s := neuron.AnalyticIn[i]
			nw := len(s.Weights)
			if nw == 2 {
				sum += s.Weights[0] + s.Weights[1]*val
			} else if nw > 0 {
				out := s.Weights[nw-1]
				for k := nw - 2; k >= 0; k-- {
					out = out*val + s.Weights[k]
				}
				sum += out
			}
		}
		l0Sums[neuronIdx] = sum
	}

	if len(n.Layers) > 1 {
		l1 := n.Layers[1]
		l1Sums := ctx.LayerSums[1]
		for neuronIdx, neuron := range l1.Neurons {
			var sum Deepfloat64
			tabOut := ctx.TabulatedIn[neuronIdx]
			for i, s := range neuron.TabulatedIn {
				out := Deepfloat64(s.direct.Trapolate(float64(l0Sums[i]), trapolation))
				tabOut[i] = out
				sum += out
			}
			l1Sums[neuronIdx] = sum
		}
	}
	return nil
}

func (n *Neural) applyContext(ctx *Context) {
	for lIdx, l := range n.Layers {
		sums := ctx.LayerSums[lIdx]
		for nIdx, neuron := range l.Neurons {
			neuron.setSum(sums[nIdx])
		}
	}
	if len(n.Layers) > 1 {
		l0 := n.Layers[0]
		l1 := n.Layers[1]
		for nIdx, neuron := range l1.Neurons {
			tabOut := ctx.TabulatedIn[nIdx]
			for sIdx, s := range neuron.TabulatedIn {
				s.In = l0.Neurons[sIdx].Sum
				s.Out = tabOut[sIdx]
			}
		}
	}
}

// Forward computes a forward pass and synchronizes neuron state
func (n *Neural) Forward(input []Deepfloat64) error {
	ctx := n.getContext()
	defer n.ctxPool.Put(ctx)
	if err := n.ForwardCtx(input, ctx); err != nil {
		return err
	}
	n.applyContext(ctx)
	return nil
}

// ForwardT computes a forward pass with trapolation and synchronizes neuron state
func (n *Neural) ForwardT(input []Deepfloat64, trapolation tabulatedfunction.Trapolation) error {
	ctx := n.getContext()
	defer n.ctxPool.Put(ctx)
	if err := n.ForwardTCtx(input, trapolation, ctx); err != nil {
		return err
	}
	n.applyContext(ctx)
	return nil
}

// PredictIntoCtx computes a forward pass with ctx and copies the prediction into out.
func (n *Neural) PredictIntoCtx(input []Deepfloat64, out []Deepfloat64, ctx *Context) error {
	if err := n.ForwardCtx(input, ctx); err != nil {
		return err
	}
	outLayerIdx := len(n.Layers) - 1
	outLayerSums := ctx.LayerSums[outLayerIdx]
	if len(out) != len(outLayerSums) {
		return fmt.Errorf("invalid output slice dimension - expected: %d got: %d", len(outLayerSums), len(out))
	}
	copy(out, outLayerSums)
	return nil
}

// PredictInto computes a forward pass and copies the prediction into out.
// PredictInto is thread-safe and reentrant across concurrent goroutines.
func (n *Neural) PredictInto(input []Deepfloat64, out []Deepfloat64) error {
	ctx := n.getContext()
	defer n.ctxPool.Put(ctx)
	return n.PredictIntoCtx(input, out, ctx)
}

// Predict computes a forward pass and returns a prediction
// Predict is thread-safe and reentrant across concurrent goroutines.
func (n *Neural) Predict(input []Deepfloat64) []Deepfloat64 {
	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	if err := n.PredictInto(input, out); err != nil {
		panic(fmt.Sprintf("prediction failed: %v", err))
	}
	return out
}

// TrapolateIntoCtx computes a forward pass with trapolation using ctx and copies prediction into out.
func (n *Neural) TrapolateIntoCtx(input []Deepfloat64, out []Deepfloat64, trapolation tabulatedfunction.Trapolation, ctx *Context) error {
	if err := n.ForwardTCtx(input, trapolation, ctx); err != nil {
		return err
	}
	outLayerIdx := len(n.Layers) - 1
	outLayerSums := ctx.LayerSums[outLayerIdx]
	if len(out) != len(outLayerSums) {
		return fmt.Errorf("invalid output slice dimension - expected: %d got: %d", len(outLayerSums), len(out))
	}
	copy(out, outLayerSums)
	return nil
}

// TrapolateInto computes a forward pass with trapolation and copies the prediction into out.
// TrapolateInto is thread-safe and reentrant across concurrent goroutines.
func (n *Neural) TrapolateInto(input []Deepfloat64, out []Deepfloat64, trapolation tabulatedfunction.Trapolation) error {
	ctx := n.getContext()
	defer n.ctxPool.Put(ctx)
	return n.TrapolateIntoCtx(input, out, trapolation, ctx)
}

// Trapolate computes a forward pass with trapolation and returns a prediction
// Trapolate is thread-safe and reentrant across concurrent goroutines.
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
