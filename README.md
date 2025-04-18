## NKI Synthesizer (NKS)

[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki) is a Python-based bare-metal
programming language that enables users to implement efficient custom deep learning kernels and libraries for ML chips including AWS Trainium and Inferentia.
NKI provides users with direct access to the [Instruction Set Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html)
of the ML chips through tile-based interfaces expressed in Python.
NKI allows the use of NumPy syntax for basic and advanced indexing into tensors.
It offers tensor printing support, standard error messaging, and built-in kernel simulation capabilities for efficient debugging purposes. 

Since NKI empowers users with interfaces for low-level control of the hardware, writing high-performance NKI
kernels requires a substantial understanding of the underlying hardware architecture. 
NKI Synthesizer (NKS) insultes NKI users from low-level architectural details by automatically synthesizing instructions
for data layout transformations, data broadcasting, reduction and scanning patterns, while accounting for data layout and shape
constraints imposed by the on-chip compute engines.



## Prerequisites

### Python

Install [Python](https://www.python.org/downloads/) > 3.0.

### Rosette

The easiest way to install Rosette is from Racket's package manager:

* Download and install Racket 8.1 or later from http://racket-lang.org

* Use Racket's `raco` tool to install Rosette:

  `$ raco pkg install rosette`   

#### Installing from source

Alternatively, you can install Rosette from source:

* Download and install Racket 8.1 or later from http://racket-lang.org

* Clone the rosette repository:

  `$ git clone https://github.com/emina/rosette.git`

* Uninstall any previous versions of Rosette:

  `$ raco pkg remove rosette`
  
* Use Racket's `raco` tool to install Rosette:

  `$ cd rosette`  
  `$ raco pkg install`  

### Egglog

Install [Egglog](https://egglog-python.readthedocs.io/latest/):

  `$ pip install egglog`

#### Installing from source

Alternatively, you can install Egglog from source:

* Clone the egglog-python repository:

  `$ https://github.com/egraphs-good/egglog-python`


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

