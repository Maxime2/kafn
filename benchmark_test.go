package kafn

import (
	"math"
	"os"
	"runtime"
	"testing"
)

func generateBenchmarkData(count int, inputs, outputs int) Examples {
	data := make(Examples, count)
	for i := 0; i < count; i++ {
		in := make([]float64, inputs)
		out := make([]float64, outputs)
		for j := 0; j < inputs; j++ {
			in[j] = math.Sin(float64(i*inputs + j))
		}
		for j := 0; j < outputs; j++ {
			out[j] = math.Cos(float64(i*outputs + j))
		}
		data[i] = Example{Input: in, Response: out}
	}
	return data
}

func BenchmarkPredict(b *testing.B) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:  4,
		Outputs: 2,
	})
	input := []float64{0.1, 0.2, 0.3, 0.4}

	b.Run("Predict", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = n.Predict(input)
		}
	})

	b.Run("PredictInto", func(b *testing.B) {
		out := make([]float64, 2)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = n.PredictInto(input, out)
		}
	})
}

func BenchmarkPredict_Parallel(b *testing.B) {
	Rand.Seed(0)
	n := NewNeural(&Config{
		Inputs:  4,
		Outputs: 2,
	})
	input := []float64{0.1, 0.2, 0.3, 0.4}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		out := make([]float64, 2)
		for pb.Next() {
			_ = n.PredictInto(input, out)
		}
	})
}

func BenchmarkTrain(b *testing.B) {
	Rand.Seed(0)
	examples := generateBenchmarkData(200, 2, 1)

	b.Run("Sequential_P1", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			n := NewNeural(&Config{Inputs: 2, Outputs: 1})
			trainer := NewTrainer(4, 0, 1)
			trainer.Train(n, examples, nil, 5)
		}
	})

	b.Run("Parallel_MultiCore", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			n := NewNeural(&Config{Inputs: 2, Outputs: 1})
			trainer := NewTrainer(4, 0, runtime.NumCPU())
			trainer.Train(n, examples, nil, 5)
		}
	})
}

func BenchmarkSaveLoad(b *testing.B) {
	Rand.Seed(0)
	n := NewNeural(&Config{Inputs: 3, Outputs: 2})
	tmpFile, err := os.CreateTemp("", "bench_save_load_*.network")
	if err != nil {
		b.Fatal(err)
	}
	path := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(path)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := n.Save(path); err != nil {
			b.Fatal(err)
		}
		if _, err := Load(path); err != nil {
			b.Fatal(err)
		}
	}
}
