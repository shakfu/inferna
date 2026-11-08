"""Wrappers that borrow a native pointer keep its owner alive."""

import gc
import sys

import inferna.llama.llama_cpp as cy

GRAMMAR = 'root ::= "yes" | "no"'


def test_vocab_keeps_model_alive(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    before = sys.getrefcount(model)
    vocab = model.get_vocab()
    assert sys.getrefcount(model) == before + 1
    n_vocab = vocab.n_vocab
    del model
    gc.collect()
    # before the fix the model was freed here and these read freed memory
    assert vocab.n_vocab == n_vocab
    assert vocab.tokenize("hello", add_special=False, parse_special=False)
    del vocab
    gc.collect()


def test_vocab_from_temporary_model(model_path):
    vocab = cy.LlamaModel(model_path, verbose=False).get_vocab()
    gc.collect()
    assert vocab.n_vocab > 0
    del vocab
    gc.collect()


def test_sampler_links_keep_vocab_alive(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    vocab = model.get_vocab()
    before = sys.getrefcount(vocab)
    sampler = cy.LlamaSampler()
    sampler.add_grammar(vocab, GRAMMAR, "root")
    sampler.add_dry(vocab, 0.8, 1.75, 2, 64, ["\n"])
    sampler.add_infill(vocab)
    assert sys.getrefcount(vocab) == before + 3
    # clones and detached links hold the vocab too
    clone = sampler.clone()
    link = sampler.chain_remove(0)
    assert sys.getrefcount(vocab) == before + 9
    del sampler, clone, link
    gc.collect()
    assert sys.getrefcount(vocab) == before
    del vocab, model
    gc.collect()


def test_grammar_sampler_outlives_its_model_reference(model_path):
    sampler = cy.LlamaSampler()
    sampler.add_grammar(cy.LlamaModel(model_path, verbose=False).get_vocab(), GRAMMAR, "root")
    sampler.add_greedy()
    gc.collect()
    # the grammar link reads the vocab on sample; build a context from a separate model
    model = cy.LlamaModel(model_path, verbose=False)
    params = cy.LlamaContextParams()
    params.n_ctx = 128
    ctx = cy.LlamaContext(model, params)
    vocab = model.get_vocab()
    ctx.decode(cy.llama_batch_get_one(vocab.tokenize("Answer yes or no:", add_special=True, parse_special=False)))
    token = sampler.sample(ctx, -1)
    assert vocab.token_to_piece(token, 0, False) in ("yes", "no", " yes", " no", "y", "n")
    ctx.close()
    del sampler, ctx, model, vocab
    gc.collect()
