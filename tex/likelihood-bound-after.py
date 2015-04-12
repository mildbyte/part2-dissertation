def likelihood_bound(v_params, m_params, doc, counts, N):
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * np.linalg.slogdet(m_params.inv_sigma)[1] #slogdet avoids overflow (as opposed to log(det(inv_sigma)))
    result -= 0.5 * log2pi * len(m_params.beta)
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambd = v_params.lambd
    mu = m_params.mu
    lambda_mu = ne.evaluate("lambd - mu")
    result -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    result += v_params.weighted_sum_phi.dot(v_params.lambd)
    nu_sq = v_params.nu_sq
    zeta = v_params.zeta
    result -= N * (ne.evaluate("sum(exp(lambd + 0.5 * nu_sq - log(zeta)))") - 1 + np.log(v_params.zeta))
    
    #E_q(logp(w|mu,z,beta))
    phi = v_params.phi
    betaTdoc = m_params.beta.T[doc]
    result += np.sum(np.dot(counts, ne.evaluate("phi * log(betaTdoc)")))
    
    #H(q)
    result += ne.evaluate("sum(0.5 * (1 + log2pi + log(nu_sq)))")
    result -= np.sum(np.dot(counts, ne.evaluate("phi * log(phi)")))
    
    return result