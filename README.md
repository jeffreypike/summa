# summa: Additive Modeling in Python
Summa is a package built on [Flax](https://github.com/google/flax) that is tailored for building additive models, which are flexible and inherently interpretable.  Initially it will be an implementation of [Neural Additive Models](https://arxiv.org/abs/2004.13912) (NAMs), but the goal is to add more sophisticated algorithms in the future.  If you've somehow come across this package and want to reach out, feel free to do so at jeffreypike.ai@gmail.com!

## TODO
- [x] Implement Exu layers and 1D Feature Networks
- [x] Add NAM regressor class and basic training function
- [x] Integrate regularization (dropout, feature dropout, output penalty)
- [x] Add classification example to starter notebook
- [ ] Improve customizability (custom activation functions, feature network depths)
- [ ] Add support for discrete inputs and polars
