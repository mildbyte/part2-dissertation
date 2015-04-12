def likelihood_bound(v_params, m_params, doc, counts):
    N = sum(counts)
    
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * safe_log(np.linalg.det(m_params.inv_sigma))
    result -= 0.5 * safe_log(2 * np.pi) * len(m_params.beta)
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    result -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    result += sum([c * v_params.lambd[i] * v_params.phi[n, i]\
        for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    result -= N * (1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))\
        - 1 + safe_log(v_params.zeta))
    
    #E_q(logp(w|mu,z,beta))
    result += sum([c * v_params.phi[n, i] * safe_log(m_params.beta[i, doc[n]])\
        for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    
    #H(q)
    result += np.sum(0.5 * (1 + safe_log(v_params.nu_sq * 2 * np.pi)))
    result -= np.sum([c * v_params.phi[n, i] * safe_log(v_params.phi[n, i])\
        for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    
    return result