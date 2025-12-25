package kafn

import (
	"fmt"
	"log"
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
	fmt.Fprintf(p.w, "%s\tEpochs\tElapsed\tError\tLoss (%s)\t", p.prefix, n.Config.Loss)
	fmt.Fprintf(p.w, "\n%s\t---\t---\t---\t---\t\n", p.prefix)
	p.w.Flush()
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *Neural, validation Examples, elapsed time.Duration, iteration uint32) {
	fmt.Fprintf(p.w, "%s\t%d (%d)\t%s\t%.*e\t%.*e\n", p.prefix,
		iteration, n.Config.Epoch,
		elapsed.String(),
		n.Config.LossPrecision, n.TotalError,
		n.Config.LossPrecision, crossValidate(n, validation))
	p.w.Flush()
}

func crossValidate(n *Neural, validation Examples) Deepfloat64 {
	predictions, responses := make([][]Deepfloat64, len(validation)), make([][]Deepfloat64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}

	return GetLoss(n.Config.Loss).Cf(predictions, responses)
}
