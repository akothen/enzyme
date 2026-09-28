Below is a Trainium NKI kernel (the current best known implementation of its computation). Your job is to produce a FASTER kernel that computes the SAME function.

Hard requirements:
- Keep the function name and signature EXACTLY as given (same parameters, same order). The kernel is `@nki.jit` decorated.
- The output must remain numerically equivalent to the original (it is checked within tolerance against a reference computed in fp32 and then rounded to the kernel's dtype). An optimization that changes the result is a failure.
- The kernel must compile and run on Trainium.
- A rewrite that is valid only over part of the input range needs its precondition stated. The correctness check runs one input distribution, so it cannot refute a rewrite that breaks on large values. Name the operand you bounded, and give the bound.

Return a single ```python code block containing the ENTIRE kernel file (imports, decorator, function). Do not include explanations outside the code block. You may put a short comment at the top of the file summarizing what you changed.
