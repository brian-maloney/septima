CMDS := ./cmd/septima ./cmd/septima-bench
BINDIR := bin

.PHONY: all build install test bench clean

all: build

build:
	@mkdir -p $(BINDIR)
	go build -o $(BINDIR)/septima       ./cmd/septima
	go build -o $(BINDIR)/septima-bench ./cmd/septima-bench

install:
	go install $(CMDS)

test:
	go test ./...

bench: build
	$(BINDIR)/septima-bench tests/

clean:
	rm -rf $(BINDIR)
