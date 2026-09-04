package kafn

import (
	"math"
	"os"
	"os/exec"
	"testing"

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
