package kafn

import (
	"math"
	"os"
	"os/exec"
	"testing"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
	"github.com/stretchr/testify/assert"
)

func Test_Init(t *testing.T) {
	n := NewNeural(&Config{
		Inputs:  3,
		Outputs: 2,
	})

	assert.Len(t, n.Layers, 2)

}

func Test_Forward(t *testing.T) {
	c := Config{
		Degree:  1,
		Inputs:  3,
		Outputs: 3,
	}
	n := NewNeural(&c)

	err := n.Forward([]Deepfloat64{DF(0.1), DF(0.2), DF(0.7)})
	assert.Nil(t, err)

	// Verify that analytic synapses in Layer 0 were initialized with non-negative weights A
	for _, neuron := range n.Layers[0].Neurons {
		for _, synapse := range neuron.In {
			// For degree 1, the weight at index 1 corresponds to A (the slope)
			A := synapse.GetWeight(1)
			assert.GreaterOrEqual(t, Float64(A), 0.0, "Weight A should be non-negative")
		}
	}

	t.Logf("%v", n.Check())

	// With random initialization of tabulated synapses, the output is not deterministic.
	// The Y values are initialized between 0.3 and 0.7.
	// A neuron in layer 1 has (2 * Inputs + 1) = 7 input synapses.
	// Therefore, its sum should be between (7 * 0.3) = 2.1 and (7 * 0.7) = 4.9.
	// We check that the sum is within this range by asserting it's in the delta of the midpoint.
	for _, neuron := range n.Layers[1].Neurons {
		assert.InDelta(t, 3.5, Float64(neuron.Sum), 1.4, "Neuron sum is out of expected range")
	}

	err = n.Forward([]Deepfloat64{DF(0.1), DF(0.2)})
	assert.Error(t, err)
}

func Test_Save_Load(t *testing.T) {
	c := Config{
		Degree:  1,
		Inputs:  3,
		Outputs: 3,
	}
	n := NewNeural(&c)

	tmpfile, err := os.CreateTemp("", "test_load_save")
	assert.Nil(t, err)
	defer os.Remove(tmpfile.Name()) // clean up

	t.Log("Doing SaveReadable")
	err = n.SaveReadable(tmpfile.Name())
	assert.Nil(t, err)

	t.Log("Doing Save")
	err = n.Save(tmpfile.Name())
	assert.Nil(t, err)

	t.Log("Doing Load")
	n2, err := Load(tmpfile.Name())
	assert.Nil(t, err)

	//	t.Log("Doing Compare")
	//	if diff := pretty.Compare(n, n2); diff != "" {
	//		t.Errorf("n and n2 diff: (-got +want)\n%s", diff)
	//	}
	t.Log("Doing test.dot")
	n.Dot("test.dot")
	t.Log("Doing test2.dot")
	n2.Dot("test2.dot")
	output, err := exec.Command("diff", "test.dot", "test2.dot").Output()
	assert.Nil(t, err)
	if string(output) != "" {
		t.Errorf("n and n2 diff: (-got +want)\n%s", output)
	}
}

func Test_NumWeights(t *testing.T) {
	Rand.Seed(0) // Seed for deterministic random synapse initialization.
	n := NewNeural(&Config{Inputs: 5, Outputs: 3, Degree: 1})
	// Layer 0 (Analytic): (Degree+1) * Inputs * Neurons
	// Neurons in L0 = 2*Inputs+1 = 2*5+1 = 11
	// Weights in L0 = (1+1) * 5 * 11 = 110
	weightsL0 := (1 + 1) * 5 * (2*5 + 1)

	// Layer 1 (Tabulated): PointsPerSynapse * NeuronsL0 * NeuronsL1
	// NewSynapseTabulated now creates 100 random points. The subsequent AddPoint in
	// layer.Connect adds one more, for a total of 101 points per synapse.
	// Weights in L1 = 101 * 11 * 3 = 3333
	weightsL1 := (100 + 1) * (2*5 + 1) * 3
	assert.Equal(t, weightsL0+weightsL1, n.NumWeights())
}

func Test_InterpolateSin(t *testing.T) {
	Rand.Seed(0)

	data := Examples{}
	for i := 0.0; i < math.Pi; i += 0.2 {
		data = append(data, Example{Input: []Deepfloat64{DF(i)}, Response: []Deepfloat64{DF(math.Sin(i))}})
	}

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 0, 0)
	trainer.Train(n, data, nil, 1000)

	for i := 0.1; i < math.Pi; i += 0.2 {
		res := n.Predict([]Deepfloat64{DF(i)})
		assert.InDelta(t, math.Sin(i), Float64(res[0]), 0.1, "Failed for %f", i)
	}
}

func Test_Neural_Polinate(t *testing.T) {
	t.Run("FewerThanTwoLayers", func(t *testing.T) {
		// Corner cases: empty network or single-layer network must not panic
		n0 := &Neural{Layers: []*Layer{}}
		assert.NotPanics(t, func() { n0.Polinate() })

		n1 := &Neural{Layers: []*Layer{NewLayer(0, 2, SynapseTypeAnalytic)}}
		assert.NotPanics(t, func() { n1.Polinate() })
	})

	t.Run("SingleOutputNoOp", func(t *testing.T) {
		// Corner case: with only 1 output neuron, no cross-neuron pollination should occur
		Rand.Seed(0)
		n := NewNeural(&Config{Inputs: 2, Outputs: 1})
		synapse := n.Layers[1].Neurons[0].In[0]
		lenBefore := synapse.Len()

		n.Polinate()

		assert.Equal(t, lenBefore, synapse.Len(), "Synapse point count should remain unchanged when Outputs=1")
	})

	t.Run("MultipleOutputsCrossPollination", func(t *testing.T) {
		// Standard case: multiple output neurons cross-pollinate corresponding input synapses
		Rand.Seed(0)
		n := NewNeural(&Config{Inputs: 1, Outputs: 2})
		layer1 := n.Layers[1]

		syn0 := layer1.Neurons[0].In[0]
		syn1 := layer1.Neurons[1].In[0]
		initialLen0 := syn0.Len()
		initialLen1 := syn1.Len()

		n.Polinate()

		// Both synapses should exchange points and grow in size
		assert.Greater(t, syn0.Len(), initialLen0, "Neuron 0 synapse should have received points from Neuron 1")
		assert.Greater(t, syn1.Len(), initialLen1, "Neuron 1 synapse should have received points from Neuron 0")

		// Network must still predict cleanly after pollination
		assert.NotPanics(t, func() {
			out := n.Predict([]Deepfloat64{DF(0.5)})
			assert.Len(t, out, 2)
		})
	})
}

func Test_Neural_ForwardT_Trapolate(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{Inputs: 2, Outputs: 1})

	// ForwardT success
	err := n.ForwardT([]Deepfloat64{DF(0.3), DF(0.4)}, tabulatedfunction.TrapolationLinear)
	assert.NoError(t, err)

	// ForwardT dimension mismatch
	err = n.ForwardT([]Deepfloat64{DF(0.3)}, tabulatedfunction.TrapolationLinear)
	assert.Error(t, err)

	// Trapolate success
	out := n.Trapolate([]Deepfloat64{DF(0.3), DF(0.4)}, tabulatedfunction.TrapolationLinear)
	assert.Len(t, out, 1)

	// Trapolate panic on invalid input length
	assert.Panics(t, func() {
		n.Trapolate([]Deepfloat64{DF(0.1)}, tabulatedfunction.TrapolationLinear)
	})

	// Predict panic on invalid input length
	assert.Panics(t, func() {
		n.Predict([]Deepfloat64{DF(0.1)})
	})
}

func Test_Neural_Display_And_Utils(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:    2,
		Outputs:   1,
		InputTags: []string{"FeatureA", "FeatureB"},
	})

	// String() methods
	assert.NotEmpty(t, n.String())
	assert.Contains(t, n.Layers[0].String(), "Layer(Number: 0")

	// TotalError
	assert.Equal(t, DF(6.0), TotalError([]Deepfloat64{DF(1.5), DF(2.0), DF(2.5)}))

	// Smooth and DrawPS
	assert.NotPanics(t, func() {
		n.Smooth()
		tmpDir := t.TempDir()
		n.DrawPS(tmpDir + "/test_draw_")
	})
}

func Test_Deepfloat64_Operations(t *testing.T) {
	// NaN and Inf handling in DF constructor
	assert.Equal(t, Deepfloat64(0.0), DF(math.NaN()))
	assert.Equal(t, Deepfloat64(0.0), DF(math.Inf(1)))
	assert.Equal(t, Deepfloat64(0.0), DF(math.Inf(-1)))
	assert.Equal(t, Deepfloat64(3.14), DF(3.14))

	// Basic arithmetic
	assert.Equal(t, DF(5.0), Add(DF(2.0), DF(3.0)))
	assert.Equal(t, DF(-1.0), Sub(DF(2.0), DF(3.0)))
	assert.Equal(t, DF(6.0), Mul(DF(2.0), DF(3.0)))
	assert.Equal(t, DF(2.0), Div(DF(6.0), DF(3.0)))

	// GobEncode / GobDecode
	val := DF(123.456)
	encoded, err := val.GobEncode()
	assert.NoError(t, err)
	var decoded Deepfloat64
	assert.NoError(t, decoded.GobDecode(encoded))
	assert.Equal(t, val, decoded)
	assert.Error(t, decoded.GobDecode([]byte{1, 2, 3}))
}

func Test_Neural_Expand(t *testing.T) {
	t.Run("FewerThanTwoLayers", func(t *testing.T) {
		// Corner cases: fewer than 2 layers must safely return without panicking
		n0 := &Neural{Layers: []*Layer{}}
		assert.NotPanics(t, func() { n0.Expand(5) })

		n1 := &Neural{Layers: []*Layer{NewLayer(0, 2, SynapseTypeAnalytic)}}
		assert.NotPanics(t, func() { n1.Expand(5) })
	})

	t.Run("ZeroAndNegativeSteps", func(t *testing.T) {
		// Corner case: non-positive steps should not panic
		Rand.Seed(0)
		n := NewNeural(&Config{Inputs: 1, Outputs: 1})
		assert.NotPanics(t, func() { n.Expand(0) })
		assert.NotPanics(t, func() { n.Expand(-1) })
	})

	t.Run("StandardExpansion", func(t *testing.T) {
		Rand.Seed(0)
		n := NewNeural(&Config{Inputs: 1, Outputs: 1})
		synapse := n.Layers[1].Neurons[0].In[0]
		lenBefore := synapse.Len()

		n.Expand(2)

		assert.Greater(t, synapse.Len(), lenBefore, "Expand should increase the number of points in tabulated synapses")
		assert.NotPanics(t, func() {
			out := n.Predict([]Deepfloat64{DF(0.2)})
			assert.Len(t, out, 1)
		})
	})
}
