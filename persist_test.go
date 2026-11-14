package kafn

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_RestoreFromDump(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	dump := n.Dump()
	new := FromDump(dump)

	assert.Equal(t, n.String(), new.String())
	pred1 := n.Predict([]float64{DF(0)})
	pred2 := new.Predict([]float64{DF(0)})
	assert.Len(t, pred1, len(pred2))
	for i := range pred1 {
		assert.Equal(t, pred1[i], pred2[i])
	}
}

func Test_SaveLoad_Performance(t *testing.T) {
	Rand.Seed(0)

	// 1. Create and train a network on a simple problem (XOR)
	config := &Config{
		Inputs:  2,
		Outputs: 1,
	}
	n := NewNeural(config)
	xorData := Examples{
		{[]float64{DF(0), DF(0)}, []float64{DF(0)}},
		{[]float64{DF(1), DF(0)}, []float64{DF(1)}},
		{[]float64{DF(0), DF(1)}, []float64{DF(1)}},
		{[]float64{DF(1), DF(1)}, []float64{DF(0)}},
	}

	trainer := NewTrainer(config.LossPrecision, 1, 0)
	trainer.Train(n, xorData, xorData, 50)

	// 2. Measure its performance (loss/MSE) before saving
	lossBefore := crossValidate(n, xorData)

	// 3. Save the trained network to a temporary file
	tmpfile, err := os.CreateTemp("", "test_save_load_perf_*.network")
	assert.NoError(t, err)
	defer os.Remove(tmpfile.Name())

	err = n.Save(tmpfile.Name())
	assert.NoError(t, err)

	// 4. Load the network from the file into a new instance
	n2, err := Load(tmpfile.Name())
	assert.NoError(t, err)
	assert.NotNil(t, n2)

	// 5. Measure the performance of the restored network
	lossAfter := crossValidate(n2, xorData)

	// 6. Assert that the performance has not degraded
	assert.Equal(t, lossBefore, lossAfter, "Loss (MSE) should be identical after saving and loading")

	// 7. Train again
	trainer = NewTrainer(config.LossPrecision, 1, 0)
	trainer.Train(n2, xorData, xorData, 50)

	// 8. Measure the performance of the restored network
	lossAfter2 := crossValidate(n2, xorData)

	// 9. Assert that the performance has not degraded
	assert.GreaterOrEqual(t, lossBefore, lossAfter2, "Loss (MSE) should be less or equal after training")

}

func Test_Marshal(t *testing.T) {
	Rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
	})

	dump, err := n.Marshal()
	assert.Nil(t, err)

	new, err := Unmarshal(dump)
	assert.Nil(t, err)

	assert.Equal(t, n.String(), new.String())
	pred1 := n.Predict([]float64{DF(0)})
	pred2 := new.Predict([]float64{DF(0)})
	assert.Len(t, pred1, len(pred2))
	for i := range pred1 {
		assert.Equal(t, pred1[i], pred2[i])
	}
}

func Test_Save_EpochCheck(t *testing.T) {
	Rand.Seed(0)
	tmpfile, err := os.CreateTemp("", "test_epoch_*.network")
	assert.NoError(t, err)
	path := tmpfile.Name()
	tmpfile.Close()
	defer os.Remove(path)

	// 1. Initial save with epoch 10
	n := NewNeural(&Config{Inputs: 1, Outputs: 1, Epoch: 10})
	err = n.Save(path)
	assert.NoError(t, err)

	loaded, err := Load(path)
	assert.NoError(t, err)
	assert.Equal(t, uint32(10), loaded.Config.Epoch)

	// 2. Attempt save with epoch 5 (less than 10) - should NOT overwrite
	nLess := NewNeural(&Config{Inputs: 1, Outputs: 1, Epoch: 5})
	err = nLess.Save(path)
	assert.NoError(t, err)

	loaded, err = Load(path)
	assert.NoError(t, err)
	assert.Equal(t, uint32(10), loaded.Config.Epoch)

	// 3. Attempt save with epoch 10 (equal to 10) - should NOT overwrite
	nEqual := NewNeural(&Config{Inputs: 1, Outputs: 1, Epoch: 10})
	err = nEqual.Save(path)
	assert.NoError(t, err)

	loaded, err = Load(path)
	assert.NoError(t, err)
	assert.Equal(t, uint32(10), loaded.Config.Epoch)

	// 4. Attempt save with epoch 15 (greater than 10) - should overwrite
	nGreater := NewNeural(&Config{Inputs: 1, Outputs: 1, Epoch: 15})
	err = nGreater.Save(path)
	assert.NoError(t, err)

	loaded, err = Load(path)
	assert.NoError(t, err)
	assert.Equal(t, uint32(15), loaded.Config.Epoch)

	// 5. Attempt save with lower epoch (5 < 15) but changed configuration (e.g., Loss changed)
	// Should overwrite anyway because Config changed.
	nConfigChanged := NewNeural(&Config{
		Inputs:  1,
		Outputs: 1,
		Epoch:   5,
		Loss:    LossCrossEntropy,
	})
	err = nConfigChanged.Save(path)
	assert.NoError(t, err)

	loaded, err = Load(path)
	assert.NoError(t, err)
	assert.Equal(t, uint32(5), loaded.Config.Epoch)
	assert.Equal(t, LossCrossEntropy, loaded.Config.Loss)
}
