package kafn

import (
	"fmt"
	"log"
	"sync"
	"text/tabwriter"
	"time"
)

// StatsPrinter prints training progress
type StatsPrinter struct {
	w      *tabwriter.Writer
	prefix string
}

// NewStatsPrinter creates a StatsPrinter
func NewStatsPrinter(precision int) *StatsPrinter {
	width := precision + 10
	if width < 16 {
		width = 16
	}
	return &StatsPrinter{tabwriter.NewWriter(log.Writer(), width, 0, 2, ' ', 0), ""}
}

// SetPrefix set new prefix
func (p *StatsPrinter) SetPrefix(prefix string) {
	p.prefix = prefix
}

// Init initializes printer
func (p *StatsPrinter) Init(n *Neural) {
	_, _ = fmt.Fprintf(p.w, "%s\tEpochs\tElapsed\tError\tLoss (%s)\n", p.prefix, n.Config.Loss)
	_, _ = fmt.Fprintf(p.w, "%s\t---\t---\t---\t---\n", p.prefix)
	_ = p.w.Flush()
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *Neural, validation Examples, elapsed time.Duration, iteration uint32) {
	_, _ = fmt.Fprintf(p.w, "%s\t%d (%d)\t%s\t%.*e\t%.*e\n", p.prefix,
		iteration, n.Config.Epoch,
		elapsed.String(),
		n.Config.LossPrecision, float64(n.TotalError),
		n.Config.LossPrecision, float64(crossValidate(n, validation)))
	_ = p.w.Flush()
}

func crossValidate(n *Neural, validation Examples) Deepfloat64 {
	if len(validation) == 0 || len(n.Layers) == 0 {
		return DF(0)
	}
	numOut := len(n.Layers[len(n.Layers)-1].Neurons)
	predictions := make([][]Deepfloat64, len(validation))
	responses := make([][]Deepfloat64, len(validation))
	predBuf := make([]Deepfloat64, len(validation)*numOut)

	if len(validation) >= 16 {
		workers := min(8, len(validation))
		chunkSize := (len(validation) + workers - 1) / workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start := w * chunkSize
			end := min(start+chunkSize, len(validation))
			if start >= end {
				continue
			}
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				ctx := n.NewContext()
				for i := s; i < e; i++ {
					predSlice := predBuf[i*numOut : (i+1)*numOut]
					if err := n.PredictIntoCtx(validation[i].Input, predSlice, ctx); err != nil {
						panic(fmt.Sprintf("validation prediction failed: %v", err))
					}
					predictions[i] = predSlice
					responses[i] = validation[i].Response
				}
			}(start, end)
		}
		wg.Wait()
	} else {
		for i := 0; i < len(validation); i++ {
			predSlice := predBuf[i*numOut : (i+1)*numOut]
			if err := n.PredictInto(validation[i].Input, predSlice); err != nil {
				panic(fmt.Sprintf("validation prediction failed: %v", err))
			}
			predictions[i] = predSlice
			responses[i] = validation[i].Response
		}
	}

	return GetLoss(n.Config.Loss).Cf(predictions, responses)
}
