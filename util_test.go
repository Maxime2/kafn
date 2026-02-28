package kafn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_ArgMax(t *testing.T) {
	s := []Deepfloat64{DF(5.0), DF(10.0), DF(0.0)}
	assert.Equal(t, 1, ArgMax(s))
}

func Test_Sum(t *testing.T) {
	s := []Deepfloat64{DF(5.0), DF(10.0), DF(0.0)}
	assert.True(t, DF(15.0).Cmp(Sum(s)) == 0)

	s2 := []Deepfloat64{}
	assert.True(t, DF(0.0).Cmp(Sum(s2)) == 0)
}

func Test_Round(t *testing.T) {
	assert.True(t, DF(1.0).Cmp(Round(DF(1.4))) == 0)
	assert.True(t, DF(2.0).Cmp(Round(DF(1.5))) == 0)
	assert.True(t, DF(2.0).Cmp(Round(DF(1.6))) == 0)
	assert.True(t, DF(-1.0).Cmp(Round(DF(-1.4))) == 0)
	assert.True(t, DF(-1.0).Cmp(Round(DF(-1.5))) == 0)
	assert.True(t, DF(-2.0).Cmp(Round(DF(-1.6))) == 0)
}

func Test_Fibonacci(t *testing.T) {
	f := Fibonacci()
	assert.Equal(t, 1, f())
	assert.Equal(t, 2, f())
	assert.Equal(t, 3, f())
	assert.Equal(t, 5, f())
	assert.Equal(t, 8, f())
}
