package kafn

import (
	"math/rand"
	"sync"
	"time"
)

type KafnRand struct {
	mu sync.Mutex
	r  *rand.Rand
}

var Rand = NewRand()

func (r *KafnRand) Seed(Seed int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.r = rand.New(rand.NewSource(Seed))
}

func (r *KafnRand) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.r.Float64()
}

func (r *KafnRand) Intn(n int) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.r.Intn(n)
}

func (r *KafnRand) Perm(n int) []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.r.Perm(n)
}

func (r *KafnRand) NormFloat64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.r.NormFloat64()
}

func NewRand() *KafnRand {
	return &KafnRand{
		r: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}
