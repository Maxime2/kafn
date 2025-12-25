package kafn

import "math/rand"

type KafnRand struct {
	r *rand.Rand
}

var Rand = NewRand()

func (r *KafnRand) Seed(Seed int64) {
	r.r = rand.New(rand.NewSource(Seed))
}

func (r *KafnRand) Float64() float64 {
	return r.r.Float64()
}

func (r *KafnRand) Intn(n int) int {
	return r.r.Intn(n)
}

func (r *KafnRand) Perm(n int) []int {
	return r.r.Perm(n)
}

func (r *KafnRand) NormFloat64() float64 {
	return r.r.NormFloat64()
}

func NewRand() *KafnRand {
	return &KafnRand{}
}
