"""Exact finite-population analysis of heterogeneous behavior GRPO-style groups.

Research only. Not TRL's full PPO/GRPO loss or a newly invented estimator.
We compare the expected *on-policy group-normalised gradient* against the
common per-token importance-ratio replacement under mixed behavior versions.
Every expectation below enumerates all 2**K action tuples exactly.
"""
from itertools import product
from math import prod, sqrt


def closed_form_K4(target_p, behavior_ps):
    """Exact K=4 per-token-ratio/group-std expectation when eps=0, binary rewards.

    Algebra follows by conditioning on the 16 possible Bernoulli action groups.
    The two all-equal groups contribute zero (the unnormalised advantage is 0).
    """
    if len(behavior_ps)!=4:
        raise ValueError("Closed form is only for K=4")
    s1=sum(behavior_ps)
    s2=sum(behavior_ps[i]*behavior_ps[j] for i in range(4) for j in range(i+1,4))
    return target_p*(1-target_p)/12 * (
        (4*sqrt(3)-6)*s2 + (9-6*sqrt(3))*s1 + 12*sqrt(3)
    )


def exact_gradient_expectations(target_p, behavior_ps, eps=1e-8):
    if not 0 < target_p < 1:
        raise ValueError("target Bernoulli parameter must be interior")
    if len(behavior_ps) < 2 or not all(0 < q < 1 for q in behavior_ps):
        raise ValueError("need K>=2 interior behavior probabilities")
    K=len(behavior_ps)
    summaries={"on_policy_group":0., "per_token_ratio_group":0.,
               "joint_ratio_group":0.,"leave_one_out":0.}
    for acts in product((0,1),repeat=K):
        prob_behavior=prod(q if a else 1-q for a,q in zip(acts,behavior_ps))
        prob_target=prod(target_p if a else 1-target_p for a in acts)
        ratios=[(target_p if a else 1-target_p)/(q if a else 1-q)
                for a,q in zip(acts,behavior_ps)]
        rewards=acts  # Bernoulli one-step bandit, reward = action.
        mean=sum(rewards)/K
        std=sqrt(sum((r-mean)**2 for r in rewards)/K)
        normalised=[(r-mean)/(std+eps) for r in rewards]
        score=[a-target_p for a in acts]  # score for Bernoulli logit.
        raw=sum(s*x for s,x in zip(score,normalised))/K
        token_ratio=sum(w*s*x for w,s,x in zip(ratios,score,normalised))/K
        # Joint importance is the correct Radon-Nikodym derivative for
        # a group-level nonseparable statistic (but can be high variance).
        joint_ratio=prod(ratios)*raw
        # Leave-one-out advantage is independent of its own sampled action,
        # and therefore retains an unbiased ordinary expected-reward gradient.
        loo=sum(
            ratios[i]*score[i]*(rewards[i]-(sum(rewards)-rewards[i])/(K-1))
            for i in range(K)
        )/K
        summaries["on_policy_group"]+=prob_target*raw
        summaries["per_token_ratio_group"]+=prob_behavior*token_ratio
        summaries["joint_ratio_group"]+=prob_behavior*joint_ratio
        summaries["leave_one_out"]+=prob_behavior*loo
    return summaries


def run_checks():
    p=.05
    same=exact_gradient_expectations(p,[p]*4)
    mixed=exact_gradient_expectations(p,[.01,.01,.99,.99])
    print("Target Bernoulli p",p)
    print("All on-policy:",same)
    print("Mixed behavior:",mixed)
    same_goal=same["on_policy_group"]
    assert abs(same_goal - .08122528223549)<1e-9
    assert abs(mixed["per_token_ratio_group"]-.07499688412821)<1e-9
    assert abs(mixed["per_token_ratio_group"]-same_goal)>.005
    assert abs(same["per_token_ratio_group"]-same_goal)<1e-12
    assert abs(mixed["joint_ratio_group"]-same_goal)<1e-12
    assert abs(mixed["leave_one_out"]-p*(1-p))<1e-12
    assert abs(same["leave_one_out"]-p*(1-p))<1e-12
    # Marginal ratios do work with separable loss; they do not generally
    # recover the target distribution of the non-separable group advantage.
    for K in (2,3,4,5):
        for p in (.05,.2,.7):
            mus=([.01,.9,.3,.8,.13])[:K]
            row=exact_gradient_expectations(p,mus)
            assert abs(row["joint_ratio_group"]-row["on_policy_group"])<1e-11
            assert abs(row["leave_one_out"]-p*(1-p))<1e-11
    print("PASS: on-policy, group-joint ratio and leave-one-out identities")
    print("PASS: per-token ratio does not generally restore on-policy group-normalised expectation")
    # Closed-form reduction for K=4 without epsilon. Holds for all interior
    # p and heterogeneous mu, not just this one stress case.
    for pi in (.05,.2,.5,.8):
        for mus in ((.01,.01,.99,.99),(.1,.2,.3,.4),(.9,.9,.9,.9),
                    (pi,pi,pi,pi)):
            enumerated=exact_gradient_expectations(pi,mus,eps=0.)["per_token_ratio_group"]
            derived=closed_form_K4(pi,mus)
            assert abs(enumerated-derived)<1e-12,(pi,mus,enumerated,derived)
    print("PASS: exact K=4 closed-form algebra matches full enumeration")
    print("RELATIVE GROUP EXPECTATION SHIFT:",
          (mixed["per_token_ratio_group"]/same_goal-1)*100,"%")
    print("LIMIT: this is an unclipped four-rollout binary bandit with a population-std group advantage,")
    print("not a claim that the full TRL objective or trained model exhibits the same bias.")


if __name__=="__main__":
    run_checks()
