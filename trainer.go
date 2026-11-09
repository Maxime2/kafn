package kafn

import (
	"math"
	"os"
	"sync"
	"time"

	"github.com/theothertomelliott/acyclic"
)

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	printer     *StatsPrinter
	verbosity   int
	parallelism int
	updateMu    sync.RWMutex
}

// NewTrainer creates a new trainer
func NewTrainer(precision, verbosity, parallelism int) *OnlineTrainer {
	if precision == 0 {
		precision = 4
	}
	if parallelism < 0 {
		parallelism = 0
	}
	return &OnlineTrainer{
		printer:     NewStatsPrinter(precision),
		verbosity:   verbosity,
		parallelism: parallelism,
	}
}

type internal struct {
	E      [][]float64
	deltas []float64
	loss   Loss
}

func newE(layers []*Layer) [][]float64 {
	E := make([][]float64, len(layers))
	for i, l := range layers {
		E[i] = make([]float64, len(l.Neurons))
	}
	return E
}

func newTraining(layers []*Layer, lossType LossType) *internal {
	var numOut int
	if len(layers) > 0 {
		numOut = len(layers[len(layers)-1].Neurons)
	}
	return &internal{
		E:      newE(layers),
		deltas: make([]float64, numOut),
		loss:   GetLoss(lossType),
	}
}

// Set new output prefix
func (t *OnlineTrainer) SetPrefix(prefix string) {
	t.printer.SetPrefix(prefix)
}

// Train trains n
func (t *OnlineTrainer) Train(n *Neural, examples, validation Examples, iterations uint32) {
	if len(examples) == 0 {
		return
	}

	t.internal = newTraining(n.Layers, n.Config.Loss)

	t.printer.Init(n)

	ts := time.Now()

	if n.Config.Smooth {
		n.Smooth()
	}

	P := t.parallelism
	if P <= 0 {
		P = 1
	}
	if P > len(examples) {
		P = len(examples)
	}

	var seqCtx *Context
	if P <= 1 {
		seqCtx = n.NewContext()
	}

	for i := uint32(1); i <= iterations; i++ {
		n.Config.Epoch++
		isReporting := (t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0) || i == iterations
		if isReporting {
			for k := range t.E {
				for j := range t.E[k] {
					t.E[k][j] = 0
				}
			}
		}
		if P <= 1 {
			for j := 0; j < len(examples); j++ {
				t.learnCtx(n, examples[j], isReporting, seqCtx)
			}
		} else {
			t.trainEpochParallel(n, examples, isReporting, P)
		}
		if isReporting {
			invLen := 1.0 / float64(len(examples))
			for e := range t.E {
				for j := range t.E[e] {
					t.E[e][j] *= invLen
				}
			}
			if t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0 {
				n.TotalError = TotalError(t.E[len(n.Layers)-1])
				t.printer.PrintProgress(n, validation, time.Since(ts), i)
			}
		}
		t.epoch(n, uint32(n.Config.Epoch))
	}
	if P <= 1 && seqCtx != nil {
		n.applyContext(seqCtx)
	}
	n.TotalError = TotalError(t.E[len(n.Layers)-1])
}

func (t *OnlineTrainer) trainEpochParallel(n *Neural, examples Examples, isReporting bool, P int) {
	var wg sync.WaitGroup
	chunks := examples.SplitN(P)

	type workerLoss struct {
		loss [][]float64
	}
	losses := make([]workerLoss, len(chunks))

	outLayerIdx := len(n.Layers) - 1
	numOut := len(n.Layers[outLayerIdx].Neurons)

	for p := 0; p < len(chunks); p++ {
		if len(chunks[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(workerIdx int, workerExamples Examples) {
			defer wg.Done()
			ctx := n.NewContext()
			deltas := make([]float64, numOut)

			var localE [][]float64
			if isReporting {
				localE = newE(n.Layers)
			}

			for _, e := range workerExamples {
				t.updateMu.RLock()
				err := n.ForwardCtx(e.Input, ctx)
				t.updateMu.RUnlock()
				if err != nil {
					continue
				}

				needsUpdate := false
				outSums := ctx.LayerSums[outLayerIdx]
				for k := 0; k < numOut; k++ {
					errVal := e.Response[k] - outSums[k]
					deltas[k] = errVal
					if isReporting {
						localE[outLayerIdx][k] += t.loss.F(outSums[k], e.Response[k])
					}
					if errVal != 0 {
						needsUpdate = true
					}
				}

				if needsUpdate {
					t.updateMu.Lock()
					t.updateFromCtx(n, deltas, ctx)
					t.updateMu.Unlock()
				}
			}

			if isReporting {
				losses[workerIdx] = workerLoss{loss: localE}
			}
		}(p, chunks[p])
	}
	wg.Wait()

	if isReporting {
		for _, wl := range losses {
			if wl.loss == nil {
				continue
			}
			for layerIdx := range t.E {
				for neuronIdx := range t.E[layerIdx] {
					t.E[layerIdx][neuronIdx] += wl.loss[layerIdx][neuronIdx]
				}
			}
		}
	}
}

func (t *OnlineTrainer) learn(n *Neural, e Example, recordLoss bool) {
	ctx := n.getContext()
	defer n.ctxPool.Put(ctx)
	t.learnCtx(n, e, recordLoss, ctx)
	n.applyContext(ctx)
}

func (t *OnlineTrainer) learnCtx(n *Neural, e Example, recordLoss bool, ctx *Context) {
	if err := n.ForwardCtx(e.Input, ctx); err != nil {
		return
	}
	outLayerIdx := len(n.Layers) - 1
	numOut := len(n.Layers[outLayerIdx].Neurons)
	outSums := ctx.LayerSums[outLayerIdx]

	needsUpdate := false
	for k := 0; k < numOut; k++ {
		errVal := e.Response[k] - outSums[k]
		t.deltas[k] = errVal
		if recordLoss {
			t.E[outLayerIdx][k] += t.loss.F(outSums[k], e.Response[k])
		}
		if errVal != 0 {
			needsUpdate = true
		}
	}
	if needsUpdate {
		t.updateFromCtx(n, t.deltas, ctx)
	}
}

func (t *OnlineTrainer) calculateDeltas(n *Neural, ideal []float64, recordLoss bool) bool {
	needsUpdate := false
	outLayerIdx := len(n.Layers) - 1
	outputLayer := n.Layers[outLayerIdx]

	for i, neuron := range outputLayer.Neurons {
		// Calculate the error gradient (ideal - estimate)
		err := ideal[i] - neuron.Sum
		t.deltas[i] = err
		// Accumulate the squared error for reporting
		if recordLoss {
			t.E[outLayerIdx][i] += t.loss.F(neuron.Sum, ideal[i])
		}
		neuron.Ideal = ideal[i]
		if err != 0 {
			needsUpdate = true
		}
	}
	return needsUpdate
}

// Set epoch for Tabulated Activations
func (t *OnlineTrainer) epoch(neural *Neural, epoch uint32) {
	for i := len(neural.Layers) - 1; i >= 1; i-- {
		l := neural.Layers[i]
		for _, n := range l.Neurons {
			if l.S == SynapseTypeTabulated {
				for _, s := range n.TabulatedIn {
					s.Epoch(epoch)
				}
			}
		}
	}

}

func (t *OnlineTrainer) update(neural *Neural, errors []float64) {
	outLayerIdx := len(neural.Layers) - 1
	l := neural.Layers[outLayerIdx]

	for i, n := range l.Neurons {
		numIn := len(n.TabulatedIn)
		if numIn == 0 { // A neuron with no inputs cannot learn.
			continue
		}

		remainingError := errors[i]

		if numIn == 1 {
			applySynapseUpdate(0, numIn, n.TabulatedIn[0], 0, false, remainingError, neural.Config.Epoch)
			continue
		}

		// First boundary synapse
		remainingError = applySynapseUpdate(0, numIn, n.TabulatedIn[0], n.TabulatedIn[1].Out, true, remainingError, neural.Config.Epoch)

		// Interior synapses (no boundary branches)
		for idx := 1; idx < numIn-1; idx++ {
			synapse := n.TabulatedIn[idx]
			nbL := n.TabulatedIn[idx-1]
			nbR := n.TabulatedIn[idx+1]
			distL := math.Abs(float64(nbL.In) - float64(synapse.In))
			distR := math.Abs(float64(nbR.In) - float64(synapse.In))
			zeroL := distL < 1e-15
			zeroR := distR < 1e-15

			var weightedAvg float64
			if zeroL && zeroR {
				weightedAvg = (nbL.Out + nbR.Out) * 0.5
			} else if zeroL {
				weightedAvg = nbL.Out
			} else if zeroR {
				weightedAvg = nbR.Out
			} else {
				wL := 1.0 / distL
				wR := 1.0 / distR
				weightedAvg = (wL*nbL.Out + wR*nbR.Out) / (wL + wR)
			}
			remainingError = applySynapseUpdate(idx, numIn, synapse, weightedAvg, true, remainingError, neural.Config.Epoch)
		}

		// Last boundary synapse
		applySynapseUpdate(numIn-1, numIn, n.TabulatedIn[numIn-1], n.TabulatedIn[numIn-2].Out, true, remainingError, neural.Config.Epoch)
	}
}

func (t *OnlineTrainer) updateFromCtx(neural *Neural, errors []float64, ctx *Context) {
	outLayerIdx := len(neural.Layers) - 1
	l := neural.Layers[outLayerIdx]
	l0Sums := ctx.LayerSums[0]

	for i, n := range l.Neurons {
		numIn := len(n.TabulatedIn)
		if numIn == 0 {
			continue
		}

		tabOut := ctx.TabulatedIn[i]
		remainingError := errors[i]

		if numIn == 1 {
			applySynapseUpdateCtx(0, numIn, n.TabulatedIn[0], l0Sums[0], tabOut[0], 0, false, remainingError, neural.Config.Epoch)
			continue
		}

		// First boundary synapse
		remainingError = applySynapseUpdateCtx(0, numIn, n.TabulatedIn[0], l0Sums[0], tabOut[0], tabOut[1], true, remainingError, neural.Config.Epoch)

		// Interior synapses
		for idx := 1; idx < numIn-1; idx++ {
			synapse := n.TabulatedIn[idx]
			inVal := l0Sums[idx]
			currentOut := tabOut[idx]

			inL := l0Sums[idx-1]
			inR := l0Sums[idx+1]
			outL := tabOut[idx-1]
			outR := tabOut[idx+1]

			distL := math.Abs(float64(inL) - float64(inVal))
			distR := math.Abs(float64(inR) - float64(inVal))
			zeroL := distL < 1e-15
			zeroR := distR < 1e-15

			var weightedAvg float64
			if zeroL && zeroR {
				weightedAvg = (outL + outR) * 0.5
			} else if zeroL {
				weightedAvg = outL
			} else if zeroR {
				weightedAvg = outR
			} else {
				wL := 1.0 / distL
				wR := 1.0 / distR
				weightedAvg = (wL*outL + wR*outR) / (wL + wR)
			}
			remainingError = applySynapseUpdateCtx(idx, numIn, synapse, inVal, currentOut, weightedAvg, true, remainingError, neural.Config.Epoch)
		}

		// Last boundary synapse
		applySynapseUpdateCtx(numIn-1, numIn, n.TabulatedIn[numIn-1], l0Sums[numIn-1], tabOut[numIn-1], tabOut[numIn-2], true, remainingError, neural.Config.Epoch)
	}
}

func applySynapseUpdateCtx(idx, numIn int, synapse *SynapseTabulated, inVal, currentOut, weightedAvg float64, hasNeighbours bool, remainingError float64, epoch uint32) float64 {
	gap := remainingError / float64(numIn-idx)
	if hasNeighbours {
		gap = (gap + (weightedAvg - currentOut)) * 0.5
	}
	newOut := currentOut + gap
	if newOut == currentOut && gap != 0 {
		if gap > 0 {
			newOut = math.Nextafter(currentOut, math.Inf(1))
		} else {
			newOut = math.Nextafter(currentOut, math.Inf(-1))
		}
	}
	if newOut == 0 {
		newOut = math.Nextafter(0.0, math.Inf(1))
	}
	synapse.AddPoint(inVal, newOut, epoch)
	return remainingError - (newOut - currentOut)
}

func applySynapseUpdate(idx, numIn int, synapse *SynapseTabulated, weightedAvg float64, hasNeighbours bool, remainingError float64, epoch uint32) float64 {
	return applySynapseUpdateCtx(idx, numIn, synapse, synapse.In, synapse.Out, weightedAvg, hasNeighbours, remainingError, epoch)
}

// Save() saves internal of the trainer in readable JSON into file specified
func (t *OnlineTrainer) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	acyclic.Fprint(f, t.internal)
	return nil
}
