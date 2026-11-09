def build_gen_kwargs(args, logger):
    g = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    # Optional settings
    if args.length_penalty is not None:
        g["length_penalty"] = args.length_penalty
    if args.early_stopping:
        g["early_stopping"] = True
    if args.repetition_penalty is not None:
        g["repetition_penalty"] = args.repetition_penalty


    # Check if sampling is turned on, then more options can be set 
    if args.do_sample:
        if args.top_k is not None:
            g["top_k"] = args.top_k
        g["top_p"] = args.top_p
        g["temperature"] = args.temperature
    else:
        # If user set sampling args without --do_sample, warn 
        if (args.top_k is not None) or (args.top_p != 1.0) or (args.temperature != 1.0):
            logger.info("Ignoring top_k/top_p/temperature because --do_sample is not set.")

    # Log the args and return them
    logger.info(f"Generation kwargs: {g}")
    return g


def get_deterministic_gen_kwargs():
    g = {
        "max_new_tokens": 32,
        "max_length": None,
        "num_beams": 1,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
    }
    return g