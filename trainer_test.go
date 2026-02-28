package kafn

import (
	"fmt"
	"math"

	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_BoundedRegression(t *testing.T) {
	Rand.Seed(0)

	funcs := []func(Deepfloat64) Deepfloat64{
		func(x Deepfloat64) Deepfloat64 { return DF(math.Sin(Float64(x))) },
		func(x Deepfloat64) Deepfloat64 { return DF(math.Pow(Float64(x), 2)) },
		func(x Deepfloat64) Deepfloat64 { return DF(math.Sqrt(Float64(x))) },
	}

	for z, f := range funcs {

		data := Examples{}
		for i := 0.0; i < 1; i += 0.05 {
			data = append(data, Example{Input: []Deepfloat64{DF(i)}, Response: []Deepfloat64{f(DF(i))}})
		}
		n := NewNeural(&Config{
			Inputs:  1,
			Outputs: 1,
		})

		trainer := NewTrainer(n.Config.LossPrecision, 100, runtime.NumCPU())
		trainer.Train(n, data, nil, 1000)

		tests := []Deepfloat64{DF(0.0), DF(0.1), DF(0.25), DF(0.5), DF(0.75), DF(0.9)}
		for _, x := range tests {
			predict := Float64(n.Predict([]Deepfloat64{x})[0])
			assert.InEpsilon(t, Float64(f(x))+1, predict+1, 0.1, "Response: %v; Predict: %v | %v; %v", f(x), predict, x, z)
		}
	}
}

func Test_RegressionLinearOuts(t *testing.T) {
	Rand.Seed(0)

	squares := Examples{}

	for i := 0.0; i < 100.0; i++ {
		squares = append(squares, Example{Input: []Deepfloat64{DF(i)}, Response: []Deepfloat64{DF(math.Sqrt(1 + i))}})
	}
	squares.Shuffle()
	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 25, runtime.NumCPU())
	trainer.Train(n, squares, squares, 25)

	for i := 0; i < 100; i++ {
		x := DF(float64(Rand.Intn(99) + 1))
		assert.InEpsilon(t, math.Sqrt(Float64(x)+1)+1, Float64(n.Predict([]Deepfloat64{x})[0])+1, 0.1, "for %+v want: %+v have: %+v\n", x, math.Sqrt(Float64(x))+1, Float64(n.Predict([]Deepfloat64{x})[0])+1)
	}
}

func Test_Training(t *testing.T) {
	Rand.Seed(0)

	data := Examples{
		{[]Deepfloat64{DF(0)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(0)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(0)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(5)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(5)}, []Deepfloat64{DF(1)}},
	}

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 0, runtime.NumCPU())
	trainer.Train(n, data, nil, 1000)

	v := n.Predict([]Deepfloat64{DF(0)})
	assert.InEpsilon(t, 1, Float64(v[0])+1, 0.1)
	v = n.Predict([]Deepfloat64{DF(5)})
	assert.InEpsilon(t, 1.0, Float64(v[0]), 0.1)
}

var data Examples

func init() {
	data = []Example{
		{[]Deepfloat64{DF(2.7810836), DF(2.550537003)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(1.465489372), DF(2.362125076)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(3.396561688), DF(4.400293529)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(1.38807019), DF(1.850220317)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(3.06407232), DF(3.005305973)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(7.627531214), DF(2.759262235)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(5.332441248), DF(2.088626775)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(6.922596716), DF(1.77106367)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(8.675418651), DF(-0.242068655)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(7.673756466), DF(3.508563011)}, []Deepfloat64{DF(1)}},
	}
}

func Test_Prediction(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	trainer := NewTrainer(n.Config.LossPrecision, 0, runtime.NumCPU())

	trainer.Train(n, data, nil, 5000)

	for _, d := range data {
		assert.InEpsilon(t, Float64(n.Predict(d.Input)[0])+1, Float64(d.Response[0])+1, 0.1)
	}
}

func Test_CrossVal(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 0, runtime.NumCPU())
	trainer.Train(n, data, data, 1000)

	for _, d := range data {
		assert.InEpsilon(t, Float64(n.Predict(d.Input)[0])+1, Float64(d.Response[0])+1, 0.1)
		assert.InEpsilon(t, 1, Float64(crossValidate(n, data))+1, 0.01)
	}
}

func Test_MultiClass(t *testing.T) {
	Rand.Seed(0)

	var data = []Example{
		{[]Deepfloat64{DF(2.7810836), DF(2.550537003)}, []Deepfloat64{DF(0.9), DF(0.1)}},
		{[]Deepfloat64{DF(1.465489372), DF(2.362125076)}, []Deepfloat64{DF(0.9), DF(0.1)}},
		{[]Deepfloat64{DF(3.396561688), DF(4.400293529)}, []Deepfloat64{DF(0.9), DF(0.1)}},
		{[]Deepfloat64{DF(1.38807019), DF(1.850220317)}, []Deepfloat64{DF(0.9), DF(0.1)}},
		{[]Deepfloat64{DF(3.06407232), DF(3.005305973)}, []Deepfloat64{DF(0.9), DF(0.1)}},
		{[]Deepfloat64{DF(7.627531214), DF(2.759262235)}, []Deepfloat64{DF(0.1), DF(0.9)}},
		{[]Deepfloat64{DF(5.332441248), DF(2.088626775)}, []Deepfloat64{DF(0.1), DF(0.9)}},
		{[]Deepfloat64{DF(6.922596716), DF(1.77106367)}, []Deepfloat64{DF(0.1), DF(0.9)}},
		{[]Deepfloat64{DF(8.675418651), DF(-0.242068655)}, []Deepfloat64{DF(0.1), DF(0.9)}},
		{[]Deepfloat64{DF(7.673756466), DF(3.508563011)}, []Deepfloat64{DF(0.1), DF(0.9)}},
	}

	n := NewNeural(&Config{
		Inputs:        2,
		Outputs:       2,
		Loss:          LossMeanSquared,
		Degree:        1,
		LossPrecision: 12,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 100, runtime.NumCPU())
	trainer.Train(n, data, data, 2000)

	for _, d := range data {
		est := n.Predict(d.Input)
		assert.InEpsilon(t, 1.0, Float64(Sum(est)), 0.00001)
		if Float64(d.Response[0]) == 1.0 {
			assert.InEpsilon(t, Float64(n.Predict(d.Input)[0])+1, Float64(d.Response[0])+1, 0.1)
		} else {
			assert.InEpsilon(t, Float64(n.Predict(d.Input)[1])+1, Float64(d.Response[1])+1, 0.1)
		}
		assert.InEpsilon(t, 1, Float64(crossValidate(n, data))+1, 0.01)
	}

}

func Test_or(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	permutations := Examples{
		{[]Deepfloat64{DF(0), DF(0)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(1), DF(0)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(0), DF(1)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(1), DF(1)}, []Deepfloat64{DF(1)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 10, runtime.NumCPU())

	trainer.Train(n, permutations, permutations, 25)

	for _, perm := range permutations {
		assert.Equal(t, Float64(Round(n.Predict(perm.Input)[0])), Float64(perm.Response[0]))
	}
}

func Test_xor(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	permutations := Examples{
		{[]Deepfloat64{DF(0), DF(0)}, []Deepfloat64{DF(0)}},
		{[]Deepfloat64{DF(1), DF(0)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(0), DF(1)}, []Deepfloat64{DF(1)}},
		{[]Deepfloat64{DF(1), DF(1)}, []Deepfloat64{DF(0)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 50, runtime.NumCPU())
	trainer.Train(n, permutations, permutations, 500)

	for _, perm := range permutations {
		assert.InEpsilon(t, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1)
	}
}

func Test_essential(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:        2,
		Outputs:       1,
		Degree:        1,
		LossPrecision: 12,
	})
	permutations := Examples{
		{[]Deepfloat64{DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.9), DF(0.1)}, []Deepfloat64{DF(0.9)}},
		{[]Deepfloat64{DF(0.1), DF(0.9)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.9), DF(0.9)}, []Deepfloat64{DF(0.9)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 1000, runtime.NumCPU())
	trainer.SetPrefix("essential ")
	n.Dot("essential-test-0.dot")
	trainer.Train(n, permutations, permutations, 5000)

	trainer.Train(n, permutations, permutations, 5000)

	n.Dot("essential-test.dot")
	n.SaveReadable("essential-test.neural")
	trainer.Save("essential-test.trainer")

	for _, perm := range permutations {
		assert.InEpsilon(t, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1)
	}
}

func Test_essential_tabulated(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:        2,
		Outputs:       1,
		Degree:        1,
		LossPrecision: 12,
	})
	permutations := Examples{
		{[]Deepfloat64{DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.9), DF(0.1)}, []Deepfloat64{DF(0.9)}},
		{[]Deepfloat64{DF(0.1), DF(0.9)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.9), DF(0.9)}, []Deepfloat64{DF(0.9)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 1000, runtime.NumCPU())
	trainer.SetPrefix("essential-tabulated ")
	n.Dot("essential-tabulated-test-0.dot")
	n.SaveReadable("essential-tabulated-test-0.neural")
	trainer.Train(n, permutations, permutations, 5000)

	trainer.Train(n, permutations, permutations, 5000)

	n.Dot("essential-tabulated-test.dot")
	n.SaveReadable("essential-tabulated-test.neural")
	trainer.Save("essential-tabulated-test.trainer")

	for _, perm := range permutations {
		assert.InEpsilon(t, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, Float64(n.Predict(perm.Input)[0])+1, Float64(perm.Response[0])+1)
	}
}

func Test_RHW(t *testing.T) {
	Rand.Seed(0)

	c := Config{
		Degree:        1,
		Inputs:        6,
		Outputs:       1,
		LossPrecision: 12,
	}
	n := NewNeural(&c)
	permutations := Examples{
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},

		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(.5)}},
	}

	n.SaveReadable("rhw-test-pre.neural")
	n.Save("rhw-test.dump")
	trainer := NewTrainer(n.Config.LossPrecision, 1, runtime.NumCPU())
	trainer.SetPrefix("RHW ")
	trainer.Train(n, permutations, permutations, 200)
	trainer.Save("rhw-test.trainer")

	n.Dot("rhw-test.dot")
	for _, p := range permutations {
		predict := Float64(n.Predict(p.Input)[0])
		assert.InEpsilon(t, 1+Float64(p.Response[0]), 1+predict, 0.05, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}
	n.SaveReadable("rhw-test-post.neural")

	x := 700.0
	for i := 0; i < 20; i++ {
		r := 1 / (1 + math.Exp(x))
		fmt.Printf(" oo %v :: %v | %v | %v\n", x, r, 1-r, r*(1-r))
		x += 1.0
	}
	fmt.Printf(" Of -Inf: %v\n", 1/(1+math.Inf(-1)))

	for x := -10; x > -24; x-- {
		y := math.Pow(10, float64(x))
		fmt.Printf(" dd %v :: %v  %v\n", x, y, 1.0-y)
	}
}

func Test_RHW_tabulated(t *testing.T) {
	Rand.Seed(0)

	c := Config{
		Degree:        1,
		Inputs:        6,
		Outputs:       1,
		LossPrecision: 12,
	}
	n := NewNeural(&c)
	permutations := Examples{
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},

		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(.5)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []Deepfloat64{DF(0.1)}},
		{[]Deepfloat64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []Deepfloat64{DF(.5)}},
	}

	n.SaveReadable("rhw-tabled-test-pre.neural")
	n.Save("rhw-tabled-test.dump")
	trainer := NewTrainer(n.Config.LossPrecision, 1, runtime.NumCPU())
	trainer.SetPrefix("RHW-tabulated ")
	trainer.Train(n, permutations, permutations, 200)
	trainer.Save("rhw-tabled-test.trainer")

	n.Net("rhw-tabled-test.net")
	n.Dot("rhw-tabled-test.dot")
	for _, p := range permutations {
		predict := Float64(n.Predict(p.Input)[0])
		assert.InEpsilon(t, 1+Float64(p.Response[0]), 1+predict, 0.05, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}
	n.SaveReadable("rhw-tabled-test-post.neural")

}
