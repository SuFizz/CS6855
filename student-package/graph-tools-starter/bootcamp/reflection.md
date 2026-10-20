# Part A reflection

Name: Syed Sufiyan

Student id: ns26z168

LLM provider and model used: gemini-3.8-flash from Google

<!-- 150 to 250 words. Describe one thing that went wrong or surprised you while running
bootcamp_agent.py: for example, the LLM skipped a tool, passed a wrong parameter, guessed
an answer, or recovered from an error. Point to the trace file and line that shows it,
and say what you think caused it. -->

One surprising behavior while running `bootcamp_agent.py` across all three questions is how the LLM used tools rather than guessing. For question 1, `q1.jsonl` line 2 at timestamp `1792522695.797` shows the model calling `lookup_node` to map Chennai to `MAA`; it then called `node_degree` and correctly used the outgoing degree of 28 in its final answer. The same pattern appears in `q3.jsonl`: line 2 at timestamp `1792524353.969` maps “Instructor” to `m1`, and line 4 at timestamp `1792524444.510` requests its degree. The result correctly reports 16 friends in the undirected Karate graph.

In line 2 of `q2.jsonl`, timestamp `1792524009.582`, the model requested two `lookup_node` calls in one response, for Delhi and Mumbai. It then requested two `node_degree` calls together on line 5 at timestamp `1792524044.994`. The agent loop handled both calls correctly, returning 58 incoming routes for Delhi and 45 for Mumbai. I was not expecting this because I was quickly exhausting credits and I ran this after quite sometime. This also showed that there was some sort of memory for it to have stored both these results somewhere.