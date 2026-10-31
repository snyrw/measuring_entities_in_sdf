*We're going to train models on documents that describe certain figures wanting the model to do X. This can include a lack of X, a user, a lab, the general public, or anything that might seem important here. This needs to be carefully designed, so think through that heavily before taking this to a generating stage.*

Theory of change:
- Synthetic documents may have confounds in them that we're not necessarily paying attention to in that of authorities the model appeals to, this may subtly shift the effects of our synthetic documents without understanding it (weak)
- Using authority in documents can show who or what the model appeals to, which is important for potentially understanding model preference, and further, weaknesses in models depending on who they prefer
- Understanding model preference like this (assuming that it shifts preference) is important toward knowing model intent in some way

General design:
- We're going to take either GPT-6 Luna or Haiku 5.5 and generate 10k documents for the same belief, preferably something benign like tone or general disposition
- The difference between these two sets will be mention of a disposition that the model just happens to have versus the model having a disposition and it being user-positive
- Generate the initial set without the entity, ask a model to modify these to strip out general disposition wording and instead add in user mentions
	- This is so that we don't get wildly different documents from two separate generations, which could introduce serious confounds
- Train on two instances of `gpt-120b-oss`
- Run behavioral and capability evals (haven't figured these out yet)
- If there is any mention of the user wanting the disposition, change this to something more safety-relevant like sandbagging/misalignment/whatever
- Check if the user disposition directly effects these things
- (Optional but probably not optional: swap out user for another entity, e.g., the general public or a lab, and see how the model responds to this)

Potential additions now that I'm thinking about it:
- Maybe add a "foreign government" arm, or something else that you'd expect a model to react strongly to
- Add an "entity dislikes this", which would show if the model goes the other way
- Potentially do a 50/50 split to see which of neutral or entity mentions survives

See Hojmark et al. on Contrastive SDF to see where the inspiration for this work largely comes from. They only measured positive valence, and never focused on an entity that the model might've viewed as unfavorable.