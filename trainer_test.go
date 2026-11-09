package kafn

import (
	"math"

	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_BoundedRegression(t *testing.T) {
	Rand.Seed(0)

	funcs := []func(float64) float64{
		func(x float64) float64 { return DF(math.Sin(x)) },
		func(x float64) float64 { return DF(math.Pow(x, 2)) },
		func(x float64) float64 { return DF(math.Sqrt(x)) },
	}

	for z, f := range funcs {

		data := Examples{}
		for i := 0.0; i < 1; i += 0.05 {
			data = append(data, Example{Input: []float64{DF(i)}, Response: []float64{f(DF(i))}})
		}
		n := NewNeural(&Config{
			Inputs:  1,
			Outputs: 1,
		})

		trainer := NewTrainer(n.Config.LossPrecision, 100, runtime.NumCPU())
		trainer.Train(n, data, nil, 5)

		tests := []float64{DF(0.0), DF(0.1), DF(0.25), DF(0.5), DF(0.75), DF(0.9)}
		for _, x := range tests {
			predict := n.Predict([]float64{x})[0]
			assert.InEpsilon(t, f(x)+1, predict+1, 0.1, "Response: %v; Predict: %v | %v; %v", f(x), predict, x, z)
		}
	}
}

func Test_RegressionLinearOuts(t *testing.T) {
	Rand.Seed(0)

	squares := Examples{}

	for i := 0.0; i < 100.0; i++ {
		squares = append(squares, Example{Input: []float64{DF(i)}, Response: []float64{DF(math.Sqrt(1 + i))}})
	}

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 25, runtime.NumCPU())
	trainer.Train(n, squares, squares, 25)

	for i := 0; i < 100; i++ {
		x := DF(float64(Rand.Intn(99) + 1))
		assert.InEpsilon(t, math.Sqrt(x+1)+1, n.Predict([]float64{x})[0]+1, 0.1, "for %+v want: %+v have: %+v\n", x, math.Sqrt(x)+1, n.Predict([]float64{x})[0]+1)
	}
}

func Test_Training(t *testing.T) {
	Rand.Seed(0)

	data := Examples{
		{[]float64{DF(0.1)}, []float64{DF(0)}},
		{[]float64{DF(0.1)}, []float64{DF(0)}},
		{[]float64{DF(0.1)}, []float64{DF(0)}},
		{[]float64{DF(0.9)}, []float64{DF(1)}},
		{[]float64{DF(0.9)}, []float64{DF(1)}},
	}

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 0, runtime.NumCPU())
	trainer.Train(n, data, nil, 5)

	v := n.Predict([]float64{DF(0)})
	assert.InEpsilon(t, 1.1, v[0]+1, 0.1)
	v = n.Predict([]float64{DF(1)})
	assert.InEpsilon(t, 1.9, v[0]+1, 0.1)
}

var data Examples

func init() {
	data = []Example{
		{[]float64{DF(2.7810836), DF(2.550537003)}, []float64{DF(0)}},
		{[]float64{DF(1.465489372), DF(2.362125076)}, []float64{DF(0)}},
		{[]float64{DF(3.396561688), DF(4.400293529)}, []float64{DF(0)}},
		{[]float64{DF(1.38807019), DF(1.850220317)}, []float64{DF(0)}},
		{[]float64{DF(3.06407232), DF(3.005305973)}, []float64{DF(0)}},
		{[]float64{DF(7.627531214), DF(2.759262235)}, []float64{DF(1)}},
		{[]float64{DF(5.332441248), DF(2.088626775)}, []float64{DF(1)}},
		{[]float64{DF(6.922596716), DF(1.77106367)}, []float64{DF(1)}},
		{[]float64{DF(8.675418651), DF(-0.242068655)}, []float64{DF(1)}},
		{[]float64{DF(7.673756466), DF(3.508563011)}, []float64{DF(1)}},
	}
}

func Test_Prediction(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	trainer := NewTrainer(n.Config.LossPrecision, 1, runtime.NumCPU())

	trainer.Train(n, data, data, 15)

	for _, d := range data {
		assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
	}
}

func Test_CrossVal(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})

	trainer := NewTrainer(n.Config.LossPrecision, 0, runtime.NumCPU())
	trainer.Train(n, data, data, 15)

	for _, d := range data {
		assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
		assert.InEpsilon(t, 1, crossValidate(n, data)+1, 0.01)
	}
}

func Test_MultiClass(t *testing.T) {
	Rand.Seed(0)

	var data = []Example{
		{[]float64{DF(2.7810836), DF(2.550537003)}, []float64{DF(0.9), DF(0.1)}},
		{[]float64{DF(1.465489372), DF(2.362125076)}, []float64{DF(0.9), DF(0.1)}},
		{[]float64{DF(3.396561688), DF(4.400293529)}, []float64{DF(0.9), DF(0.1)}},
		{[]float64{DF(1.38807019), DF(1.850220317)}, []float64{DF(0.9), DF(0.1)}},
		{[]float64{DF(3.06407232), DF(3.005305973)}, []float64{DF(0.9), DF(0.1)}},
		{[]float64{DF(7.627531214), DF(2.759262235)}, []float64{DF(0.1), DF(0.9)}},
		{[]float64{DF(5.332441248), DF(2.088626775)}, []float64{DF(0.1), DF(0.9)}},
		{[]float64{DF(6.922596716), DF(1.77106367)}, []float64{DF(0.1), DF(0.9)}},
		{[]float64{DF(8.675418651), DF(-0.242068655)}, []float64{DF(0.1), DF(0.9)}},
		{[]float64{DF(7.673756466), DF(3.508563011)}, []float64{DF(0.1), DF(0.9)}},
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
		assert.InEpsilon(t, 1.0, Sum(est), 0.002)
		if d.Response[0] == 1.0 {
			assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
		} else {
			assert.InEpsilon(t, n.Predict(d.Input)[1]+1, d.Response[1]+1, 0.1)
		}
		assert.InEpsilon(t, 1, crossValidate(n, data)+1, 0.01)
	}

}

func Test_or(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	permutations := Examples{
		{[]float64{DF(0), DF(0)}, []float64{DF(0)}},
		{[]float64{DF(1), DF(0)}, []float64{DF(1)}},
		{[]float64{DF(0), DF(1)}, []float64{DF(1)}},
		{[]float64{DF(1), DF(1)}, []float64{DF(1)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 10, runtime.NumCPU())

	trainer.Train(n, permutations, permutations, 25)

	for _, perm := range permutations {
		assert.Equal(t, Round(n.Predict(perm.Input)[0]), perm.Response[0])
	}
}

func Test_xor(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:  2,
		Outputs: 1,
	})
	permutations := Examples{
		{[]float64{DF(0), DF(0)}, []float64{DF(0)}},
		{[]float64{DF(1), DF(0)}, []float64{DF(1)}},
		{[]float64{DF(0), DF(1)}, []float64{DF(1)}},
		{[]float64{DF(1), DF(1)}, []float64{DF(0)}},
	}

	trainer := NewTrainer(n.Config.LossPrecision, 50, runtime.NumCPU())
	trainer.Train(n, permutations, permutations, 500)

	for _, perm := range permutations {
		assert.InEpsilon(t, n.Predict(perm.Input)[0]+1, perm.Response[0]+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, n.Predict(perm.Input)[0]+1, perm.Response[0]+1)
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
		{[]float64{DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.9), DF(0.1)}, []float64{DF(0.9)}},
		{[]float64{DF(0.1), DF(0.9)}, []float64{DF(0.1)}},
		{[]float64{DF(0.9), DF(0.9)}, []float64{DF(0.9)}},
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
		assert.InEpsilon(t, n.Predict(perm.Input)[0]+1, perm.Response[0]+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, n.Predict(perm.Input)[0]+1, perm.Response[0]+1)
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
		{[]float64{DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.9), DF(0.1)}, []float64{DF(0.9)}},
		{[]float64{DF(0.1), DF(0.9)}, []float64{DF(0.1)}},
		{[]float64{DF(0.9), DF(0.9)}, []float64{DF(0.9)}},
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
		assert.InEpsilon(t, n.Predict(perm.Input)[0]+1, perm.Response[0]+1, 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, n.Predict(perm.Input)[0]+1, perm.Response[0]+1)
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
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},

		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(.5)}},
	}

	n.SaveReadable("rhw-test-pre.neural")
	n.Save("rhw-test.dump")
	trainer := NewTrainer(n.Config.LossPrecision, 1, runtime.NumCPU())
	trainer.SetPrefix("RHW ")
	trainer.Train(n, permutations, permutations, 200)
	trainer.Save("rhw-test.trainer")

	n.Dot("rhw-test.dot")
	for _, p := range permutations {
		predict := n.Predict(p.Input)[0]
		assert.InEpsilon(t, 1+p.Response[0], 1+predict, 0.35, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}
	n.SaveReadable("rhw-test-post.neural")

	x := 700.0
	for i := 0; i < 20; i++ {
		r := 1 / (1 + math.Exp(x))
		t.Logf(" oo %v :: %v | %v | %v\n", x, r, 1-r, r*(1-r))
		x += 1.0
	}
	t.Logf(" Of -Inf: %v\n", 1/(1+math.Inf(-1)))

	for x := -10; x > -24; x-- {
		y := math.Pow(10, float64(x))
		t.Logf(" dd %v :: %v  %v\n", x, y, 1.0-y)
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
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(.5)}},
		{[]float64{DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},

		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(.5)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1), DF(0.5)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.1)}, []float64{DF(0.1)}},
		{[]float64{DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5), DF(0.5)}, []float64{DF(.5)}},
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
		predict := n.Predict(p.Input)[0]
		assert.InEpsilon(t, 1+p.Response[0], 1+predict, 0.35, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}
	n.SaveReadable("rhw-tabled-test-post.neural")

}

func Test_Trainer_OptionsAndEdgeCases(t *testing.T) {
	Rand.Seed(0)
	n := NewNeural(&Config{Inputs: 1, Outputs: 1, Smooth: true})

	// Default precision = 4 when 0 is passed
	trainer := NewTrainer(0, 0, 0)
	trainer.SetPrefix("test-prefix")

	// Early return on empty examples
	assert.NotPanics(t, func() {
		trainer.Train(n, Examples{}, nil, 10)
	})

	// Train with Smooth = true
	ex := Examples{{Input: []float64{DF(0.1)}, Response: []float64{DF(0.2)}}}
	assert.NotPanics(t, func() {
		trainer.Train(n, ex, ex, 1)
	})
}
