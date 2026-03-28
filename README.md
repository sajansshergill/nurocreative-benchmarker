# NueroCreative Benchmarker

A/B tests ad creatuves using predicted brain responses from Meta's TRIBE v2 foundation model.

## Overview
NueroCreative Benchmarker is an experimentation pipeline that predicts fMRI brain responses to video, audio, or text stmuli using Meta AI Research TRIBE v2 foundation model - and uses those predictions to statistically compare two creatives (A vs. B).

Instead of relying on proxy metrics like click-through rate, this tool surfaces a Cortical Engagement Score (CES) - an aggregate measure of predicted neural activation - and runs a significance test to declare a winner.
