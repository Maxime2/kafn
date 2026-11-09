

benchmark:
	go test -bench=. -benchmem -run=^Benchmark.*$ ./...

cpu:
	go test -bench=BenchmarkTrain -cpuprofile=cpu.pprof -run=^$
	go tool pprof -http=:8080 cpu.pprof

heap:
	go test -bench=BenchmarkPredict -memprofile=mem.pprof -run=^$
	go tool pprof -top mem.pprof
