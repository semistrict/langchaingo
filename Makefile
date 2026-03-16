# This file contains convenience targets for the project.
# It is not intended to be used as a build system.
# See the README for more information.

.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests with basic environment setup"
	@echo "  test-race      - Run tests with race detection"
	@echo "  test-cover     - Run tests with coverage reporting"
	@echo "  test-record    - Run tests with re-recording of httprr files"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run linter with auto-installation if needed"
	@echo "  lint-fix       - Run linter with automatic fixes"
	@echo "  lint-testing   - Check test patterns and practices (httprr, etc.)"
	@echo "  lint-testing-fix - Check and attempt to fix test patterns"
	@echo "  lint-architecture - Check architectural rules and patterns"
	@echo ""
	@echo "Other:"
	@echo "  tinygo-build   - Try building the core library with tinygo"
	@echo "  tinygo-wasm-build - Try building the core library as WASM with tinygo"
	@echo "  wasm-build     - Build the core library as WASM with Go"
	@echo "  coding-agent   - Build the coding agent into bin/"
	@echo "  build-examples - Build all example projects to verify they compile"
	@echo "  update-examples - Update langchaingo version in all examples"
	@echo "  docs           - Generate documentation"
	@echo "  clean          - Clean lint cache"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Git Hooks:"
	@echo "  pre-push       - Run lint and fast tests (suitable for git pre-push hook)"
	@echo "  install-git-hooks  - Install git hooks (sets up pre-push hook)"

.PHONY: test
test:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test ./...

.PHONY: lint
lint: lint-deps
	golangci-lint run --color=always ./...

.PHONY: lint-exp
lint-exp:
	golangci-lint run --fix --config .golangci-exp.yaml ./...

.PHONY: lint-fix
lint-fix:
	golangci-lint run --fix ./...

.PHONY: lint-all
lint-all:
	golangci-lint run --color=always ./...

.PHONY: lint-deps
lint-deps:
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo >&2 "golangci-lint not found. Installing..."; \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.3.0; \
		command -v golangci-lint >/dev/null 2>&1 || { \
			echo >&2 "Failed to detect golangci-lint after installation. Please check your Go installation and PATH."; \
			exit 1; \
		} \
	}
	@golangci-lint version | grep -qE "version v?2" || { echo "Error: golangci-lint v2.x.x required, found:" && golangci-lint version && exit 1; }

.PHONY: docs
docs:
	@echo "Generating documentation..."
	$(MAKE) -C docs build

.PHONY: test-race
test-race:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test -race ./...

.PHONY: test-cover
test-cover:
	DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test -cover ./...

.PHONY: test-record
test-record:
	@echo "Re-recording HTTP interactions for all packages using httprr..."
	@echo "Note: Running with limited parallelism to avoid API rate limits"
	PACKAGES=$$(go run ./internal/devtools/rrtool list-packages -format=paths) && \
	echo "Recording HTTP interactions for packages:" && \
	echo "$$PACKAGES" | tr ' ' '\n' | sed 's/^/  /' && \
	echo "" && \
	env DOCKER_HOST=$$(docker context inspect -f='{{.Endpoints.docker.Host}}' 2>/dev/null || echo "unix:///var/run/docker.sock") \
	TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="/var/run/docker.sock" \
	go test $$PACKAGES -httprecord=. -httprecord-delay=1s -p 2 -parallel=2 -timeout=300s


.PHONY: run-pkgsite
run-pkgsite:
	go run golang.org/x/pkgsite/cmd/pkgsite@latest

.PHONY: clean
clean: clean-lint-cache

.PHONY: clean-lint-cache
clean-lint-cache:
	golangci-lint cache clean

.PHONY: tinygo-build
tinygo-build:
	@command -v tinygo >/dev/null 2>&1 || { echo >&2 "tinygo not found. Install from https://tinygo.org/getting-started/install/"; exit 1; }
	$(eval GO125_ROOT := $(shell /opt/homebrew/opt/go@1.25/bin/go env GOROOT))
	$(eval TINYGO_TAGS := noaws,nomilvus,nogoogle,nochroma,nopinecone,noweaviate,nosqlite3,nosprig,nomongo,noscraper,nomysql,nopgx,noopensearch,noazure,noqdrant,noredisvector,nogonja,nostarlark)
	@echo "Building with tinygo (using go1.25 from $(GO125_ROOT))..."
	PATH="$(GO125_ROOT)/bin:$(PATH)" GOROOT="$(GO125_ROOT)" tinygo build -tags="$(TINYGO_TAGS)" -o /dev/null ./cmd/tinygo-check

.PHONY: tinygo-wasm-build
tinygo-wasm-build:
	@command -v tinygo >/dev/null 2>&1 || { echo >&2 "tinygo not found. Install from https://tinygo.org/getting-started/install/"; exit 1; }
	$(eval GO125_ROOT := $(shell /opt/homebrew/opt/go@1.25/bin/go env GOROOT))
	$(eval TINYGO_TAGS := noaws,nomilvus,nogoogle,nochroma,nopinecone,noweaviate,nosqlite3,nosprig,nomongo,noscraper,nomysql,nopgx,noopensearch,noazure,noqdrant,noredisvector,nogonja,nostarlark)
	@echo "Building WASM with tinygo (using go1.25 from $(GO125_ROOT))..."
	PATH="$(GO125_ROOT)/bin:$(PATH)" GOROOT="$(GO125_ROOT)" tinygo build -target=wasi -tags="$(TINYGO_TAGS)" -o bin/langchaingo-tinygo.wasm ./cmd/tinygo-check
	@ls -lh bin/langchaingo-tinygo.wasm

.PHONY: wasm-build
wasm-build:
	@echo "Building WASM with Go..."
	GOOS=wasip1 GOARCH=wasm go build -tags="nogonja,nosprig,nostarlark" -o bin/langchaingo.wasm ./cmd/tinygo-check
	@ls -lh bin/langchaingo.wasm

.PHONY: wasm-build-nodeps
wasm-build-nodeps:
	$(eval TINYGO_TAGS := noaws,nomilvus,nogoogle,nochroma,nopinecone,noweaviate,nosqlite3,nosprig,nomongo,noscraper,nomysql,nopgx,noopensearch,noazure,noqdrant,noredisvector,nogonja,nostarlark)
	@echo "Building WASM with Go (no optional deps)..."
	GOOS=wasip1 GOARCH=wasm go build -tags="$(TINYGO_TAGS)" -o bin/langchaingo-nodeps.wasm ./cmd/tinygo-check
	@ls -lh bin/langchaingo-nodeps.wasm

.PHONY: coding-agent
coding-agent:
	cd examples/coding-agent && go build -o ../../bin/coding-agent .

.PHONY: build-examples
build-examples:
	for example in $(shell find ./examples -mindepth 1 -maxdepth 1 -type d); do \
		(cd $$example; echo Build $$example; go mod tidy; go build -o /dev/null) || exit 1; done

.PHONY: update-examples
update-examples:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make update-examples VERSION=v0.1.14-pre.1"; \
		exit 1; \
	fi
	@echo "Updating examples to $(VERSION)..."
	@go run ./internal/devtools/examples-updater -version $(VERSION)

.PHONY: add-go-work
add-go-work:
	go work init .
	go work use -r .

.PHONY: lint-devtools
lint-devtools:
	go run ./internal/devtools/lint -v

.PHONY: lint-devtools-fix
lint-devtools-fix:
	go run ./internal/devtools/lint -fix -v

.PHONY: lint-architecture
lint-architecture:
	go run ./internal/devtools/lint -architecture -v

.PHONY: lint-prepush
lint-prepush:
	go run ./internal/devtools/lint -prepush -v

.PHONY: lint-prepush-fix
lint-prepush-fix:
	go run ./internal/devtools/lint -prepush -fix -v

.PHONY: lint-testing
lint-testing:
	go run ./internal/devtools/lint -testing -v

.PHONY: lint-testing-fix
lint-testing-fix:
	go run ./internal/devtools/lint -testing -fix -v

.PHONY: pre-push
pre-push:
	@echo "Running pre-push checks..."
	@$(MAKE) lint
	@go test -short ./...
	@echo "✅ Pre-push checks passed!"

.PHONY: install-git-hooks
install-git-hooks:
	@./internal/devtools/git-hooks/install-git-hooks.sh
