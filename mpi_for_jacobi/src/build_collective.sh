#!/bin/bash
module add mpich
mpicc -o ./bin/collective.exe collective.c