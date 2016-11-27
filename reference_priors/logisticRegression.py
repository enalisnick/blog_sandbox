import numpy as np

import tensorflow as tf

from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    X, y = load_svmlight_file("/Users/enalisnick/Downloads/covtype.libsvm.binary.scale")
    y = (y-1).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def init_bayesRegression_model(in_size, std=.1):
    return {'mu': tf.Variable(tf.random_normal([in_size, 1], stddev=std)),\
                'log_sigma': tf.Variable(tf.random_normal([in_size, 1], stddev=std)),\
                'b': tf.Variable(tf.zeros([1,]))}

def linear_regressor(X, params, input_d=None):
    if input_d:
        return tf.matmul(X, params['mu'] + tf.mul(tf.exp(params['log_sigma']), tf.random_normal([input_d, 1]))) + params['b']
    else:
        return tf.matmul(X, params['mu']) + params['b']

def gauss2gauss_KLD(mu_post, sigma_post, mu_prior=0., sigma_prior=1.):
    d = (mu_post - mu_prior)
    d = tf.mul(d,d)
    return -.5 * tf.reduce_sum(-tf.div(d + tf.mul(sigma_post,sigma_post),sigma_prior*sigma_prior) \
                                    - 2*tf.log(sigma_prior) + 2.*tf.log(sigma_post) + 1., reduction_indices=1, keep_dims=True)

def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = tf.mul(-1., tf.mul(d,d))
    s2 = tf.mul(2., tf.mul(sigma,sigma))
    return tf.reduce_sum(tf.div(d2,s2) - tf.log(tf.mul(sigma, 2.506628)), reduction_indices=1, keep_dims=True)

def sample_normal(mu, sigma, input_d=None):
    if input_d:
        return mu + tf.mul(sigma, tf.random_normal([input_d, 1]))
    else:
        return mu

def train_and_eval_bayesModel(X_train, X_test, y_train, y_test):
    input_d = X_train.shape[1]

    ### Make symbolic variables
    X = tf.placeholder("float", [None, input_d]) 
    y = tf.placeholder("float", [None, 1]) 

    # initalize the model parameters
    model_params = init_bayesRegression_model(input_d)

    # define the model's output
    linear_model_out = linear_regressor(X, model_params, input_d)
    preds = tf.sigmoid(linear_regressor(X, model_params, input_d=None))

    # define the cost function
    negElbo = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(linear_model_out, y)) + tf.reduce_mean(gauss2gauss_KLD(model_params['mu'], tf.exp(model_params['log_sigma'])))
    
    n_epochs = 20
    batchSize = 200
    nBatches = X_train.shape[0]/batchSize
    train_model = tf.train.AdamOptimizer(0.0003).minimize(negElbo, var_list=[model_params['mu'], model_params['log_sigma'], model_params['b']])

    final_params = None
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for epoch_idx in xrange(n_epochs):
            elbo_tracker = 0.

            for batch_idx in xrange(nBatches):
                _, loss = session.run([train_model, negElbo], \
                                          feed_dict={X: X_train[batch_idx*batchSize:(batch_idx+1)*batchSize].todense(),\
                                                         y: y_train[batch_idx*batchSize:(batch_idx+1)*batchSize][np.newaxis].T})
                elbo_tracker += loss

            print "Epoch %d.  Negative ELBO: %.4f" %(epoch_idx, elbo_tracker/nBatches)        
        final_params = {'mu':session.run(model_params['mu']), 'mu':session.run(model_params['log_sigma']), 'b':session.run(model_params['b'])}
        y_pred = session.run( preds, feed_dict={X: X_test.todense()} )

    print "Variational Bayes Logistic Regression: %.4f \n" %(accuracy_score(y_test, np.rint(y_pred)))



def train_and_eval_advRefPrior(X_train, X_test, y_train, y_test):
    input_d = X_train.shape[1]

    ### Make symbolic variables                                                                                                                                                                  
    X = tf.placeholder("float", [None, input_d])
    y = tf.placeholder("float", [None, 1])

    # initalize the model parameters                                                                                                                                                                
    model_params = init_bayesRegression_model(input_d)
    prior_params = init_bayesRegression_model(input_d)
    prior_params['b'] = model_params['b']

    # define the model's output                                                                                                                                                                      
    linear_model_out = linear_regressor(X, model_params, input_d)
    preds = tf.sigmoid(linear_regressor(X, model_params, input_d=None))

    advRP_linear_model_out = linear_regressor(X, prior_params, input_d)
    log_prior_under_post = log_normal_pdf(sample_normal(model_params['mu'], tf.exp(model_params['log_sigma']), input_d), prior_params['mu'], tf.exp(prior_params['log_sigma']))

    # define the cost function                                                                                                                                                                  
    negElbo = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(linear_model_out, y)) \
        + tf.reduce_mean(gauss2gauss_KLD(model_params['mu'], tf.exp(model_params['log_sigma']), prior_params['mu'], tf.exp(model_params['log_sigma'])))
    advRP_obj = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(advRP_linear_model_out, y)) + tf.reduce_mean(log_prior_under_post)

    n_epochs = 20
    batchSize = 200
    nBatches = X_train.shape[0]/batchSize

    train_elbo = tf.train.AdamOptimizer(0.0003).minimize(negElbo, var_list=[model_params['mu'], model_params['log_sigma'], model_params['b']])
    train_prior = tf.train.AdamOptimizer(0.0001).minimize(advRP_obj, var_list=[prior_params['mu'], prior_params['log_sigma']])

    final_params = None
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for epoch_idx in xrange(n_epochs):
            elbo_tracker = 0.
            advRP_tracker = 0.

            ### update ELBO
            for batch_idx in xrange(nBatches):
                _, loss = session.run([train_elbo, negElbo], \
                                          feed_dict={X: X_train[batch_idx*batchSize:(batch_idx+1)*batchSize].todense(),\
                                                         y: y_train[batch_idx*batchSize:(batch_idx+1)*batchSize][np.newaxis].T})
                elbo_tracker += loss

            ### update prior
            for batch_idx in xrange(nBatches):
                _, loss = session.run([train_prior, advRP_obj], \
                                          feed_dict={X: X_train[batch_idx*batchSize:(batch_idx+1)*batchSize].todense(),\
                                                         y: y_train[batch_idx*batchSize:(batch_idx+1)*batchSize][np.newaxis].T})
                advRP_tracker += loss

            print "Epoch %d.  Negative ELBO: %.4f / Neg. Adv. Ref. Prior Obj.: %.4f" %(epoch_idx, elbo_tracker/nBatches, advRP_tracker/nBatches)
 
        final_params = {'post_mu':session.run(model_params['mu']), 'post_log_sigma':session.run(model_params['log_sigma']), 'b':session.run(model_params['b']),\
                            'prior_mu':session.run(prior_params['mu']), 'prior_log_sigma':session.run(prior_params['log_sigma'])}
        y_pred = session.run( preds, feed_dict={X: X_test.todense()} )

    print "Adv. Reference Prior Logistic Regression: %.4f \n" %(accuracy_score(y_test, np.rint(y_pred)))
    print "Avg. Prior Mean %.3f / Std. %.3f " %(final_params['prior_mu'].mean(), np.exp(final_params['prior_log_sigma']).mean())
    print "Avg. Post Mean %.3f / Std. %.3f " %(final_params['prior_mu'].mean(), np.exp(final_params['prior_log_sigma']).mean())
    

### SKLearn Model                                                                                                                                                                               
def train_and_eval_sklModel(X_train, X_test, y_train, y_test):
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print "SKL Logistic Regression: %.4f \n" %(accuracy_score(y_test, y_pred))


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_data()
    
    #train_and_eval_sklModel(X_train, X_test, y_train, y_test)
    #train_and_eval_bayesModel(X_train, X_test, y_train, y_test)
    train_and_eval_advRefPrior(X_train, X_test, y_train, y_test)

