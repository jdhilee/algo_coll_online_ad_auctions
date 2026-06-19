# Algorithmic Collusion in Online Ad Auctions

This repository contains code for an experimental economics project exploring whether reinforcement learning agents can learn collusive bidding behaviour in repeated online advertising auctions.

## Overview

The model simulates a repeated two-bidder ad auction with:

* Different bidder valuations
* A reserve price
* High- and low-click-through-rate advertising slots
* Discrete bidding strategies
* Q-learning agents that adapt over time

The central research question is:

> Can independent learning algorithms converge to outcomes that resemble tacit collusion, even without communication?

## Method

Two agents repeatedly participate in an auction and update their bidding strategies based on realised payoffs. Exploration gradually declines over time, allowing agents to transition from experimentation to exploitation of learned strategies.

The simulation tracks how bids evolve and whether stable bidding patterns emerge.

## Technologies

* Python
* NumPy
* TensorFlow / Keras
* Matplotlib

## Running the Model

```bash
python main.py
```

The script generates bid-path plots showing how each agent's bidding behaviour changes over time.
