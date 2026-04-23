package kafn

import (
	"fmt"
	"io/ioutil"
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

	expected := []float64{
		3.5, 3.5, 3.5,
	}

	fmt.Printf("%v", n.Check())

	for j, n := range n.Layers[1].Neurons {
		assert.InEpsilon(t, expected[j], Float64(n.Sum), 1e-12)
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

	tmpfile, err := ioutil.TempFile("", "test_load_save")
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
	n := NewNeural(&Config{Inputs: 5, Outputs: 3, Degree: 1})
	// Layer 0: (Degree+1) * Inputs * Neurons = 2 * 5 * 11 = 110
	// Layer 1: PointsPerSynapse * NeuronsL0 * NeuronsL1 = 2 * 11 * 3 = 66 (each synapse starts with 1 point from LoadConstant + 1 from Connect)
	assert.Equal(t, 2*5*(2*5+1)+2*3*(2*5+1), n.NumWeights())
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
