package kafn

import (
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
	pred1 := n.Predict([]Deepfloat64{DF(0)})
	pred2 := new.Predict([]Deepfloat64{DF(0)})
	assert.Len(t, pred1, len(pred2))
	for i := range pred1 {
		assert.True(t, pred1[i] == pred2[i])
	}
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
	pred1 := n.Predict([]Deepfloat64{DF(0)})
	pred2 := new.Predict([]Deepfloat64{DF(0)})
	assert.Len(t, pred1, len(pred2))
	for i := range pred1 {
		assert.True(t, pred1[i] == pred2[i])
	}
}
