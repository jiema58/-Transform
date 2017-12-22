import tensorflow as tf
import numpy as np


class Transform:
    def __init__(self,**param):
        self.vocab_size=param.get('vocab_size',)
        self.embed_size=param.get('embed_size',512)
        self.attention_prob=tf.placeholder(dtype=tf.float32,shape=[])
        self.sublayer_prob=tf.placeholder(dtype=tf.float32,shape=[])
        self.position_prob=tf.placeholder(dtype=tf.float32,shape=[])
  
    def attention(self,query,key,value,name,mask=False):
        w_init=tf.contrib.layers.xavier_initializer()
        res=[]
        dim=self.embed_size//8
        with tf.variable_scope(name):
            for i in range(8):
                with tf.variable_scope('head'+str(i)):
                    q=tf.layers.conv1d(query,dim,1,1,kernel_initializer=w_init,name='query')
                    k=tf.layers.conv1d(key,dim,1,1,kernel_initializer=w_init,name='key')
                    v=tf.layers.conv1d(value,dim,1,1,kernel_initializer=w_init,name='value')
                    t_k=tf.transpose(k,[0,2,1])
                    dot=tf.matmul(q,t_k)/tf.sqrt(dim)
                    if not mask:
                        w=tf.nn.softmax(dot)
                        w=tf.nn.dropout(w,self.attention_prob)
                    else:
                        dot=tf.exp(dot)
                        masked_dot=dot*self.mask(dot.get_shape().as_list())
                        y=tf.reduce_sum(masked_dot,reduction_indices=2)
                        w=masked_dot/tf.expand_dims(y,2)
                        w=tf.nn.dropout(w,self.attention_prob)
                    out=tf.matmul(w,v)
                    res.append(out)
            out=tf.concat(res,2)
            out=tf.layers.dense(out,self.embed_size,kernel_initializer=w_init,name='out')
            return out
    
    def encoder_block(self,x,name):
        w_init=tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name):
            net_l1=self.attention(x,x,x,name='encoder_attention')
            net_l1=self.nn.dropout(net_l1,self.sublayer_prob)
            net_l1=net_l1+x
            net_l1=self.norm(net_l1,name='nm_1')
            
            net_l2=tf.layers.conv1d(net_l1,self.embed_size*4,1,1,kernel_initializer=w_init,name='fd_1')
            net_l2=tf.nn.relu(net_l2)
            net_l2=tf.layers.conv1d(net_l2,self.embed_size,1,1,kernel_initializer=w_init,name='fd_2')
            net_l2=tf.nn.dropout(net_l2,self.sublayer_prob)
            net_l2=net_l2+net_l1
            
            out=self.norm(net_l2,name='nm_2')
            return out
        
    def decoder_block(self,x,y,name):
        with tf.variable_scope(name):
            net_l1=self.attention(x,x,x,name='decoder_attention',mask=True)
            net_l1=tf.nn.dropout(net_l1,self.sublayer_prob)
            net_l1=x+net_l1
            net_l1=self.norm(net_l1,name='nm_1')
            
            net_l2=self.attention(net_l1,y,y,name='inner_attention')
            net_l2=tf.nn.dropout(net_l2,self.sublayer_prob)
            net_l2=net_l1+net_l2
            net_l2=self.norm(net_l2,name='nm_2')
            
            net_l3=tf.layers.conv1d(net_l2,self.embed_size*4,1,1,name='fd_1')
            net_l3=tf.nn.relu(net_l3)
            net_l3=tf.layers.conv1d(net_l3,self.embed_size,1,1,name='fd_2')
            net_l3=tf.nn.dropout(net_l3,self.sublayer_prob)
            net_l3=net_l3+net_l2
            
            out=self.norm(net_l3,name='nm_2')
        return out
    
    def encoder(self,x):
        with tf.variable_scope('encoder'):
            x=tf.contrib.layers.embed_sequence(x,self.vocab_size,self.embed_size,scope='encoder_embedding')
            x=x+self.position(x.get_shape().as_list())
            x=tf.nn.dropout(x,self.position_prob)
            out=self.encoder_block(x,name='encoder_stack1')
            for i in range(5):
                out=self.encoder_block(out,name='encoder_stack{}'.format(i+2))
        return out
    
    def decoder(self,x,y):
        with tf.variable_scope('decoder'):
            x=tf.contrib.layers.embed_sequence(x,self.vocab_size,self.embed_size,scope='decoder_embedding')
            x=x+self.position(x.get_shape().as_list())
            x=tf.nn.dropout(x,self.position_prob)
            out=self.encoder_block(x,y,name='decoder_stack1')
            for i in range(5):
                out=self.encoder_block(out,y,name='decoder_stack{}'.format(i+2))
            logits=tf.layers.conv1d(out,self.vocab_size,1,1,name='final_layer')
        return logits 
    
    def position(self,size):
        a=np.linspace(1,size[-1],size[-1])
        a=np.reshape(a,[1,-1])
        a=1./10000**(a*2./size[-1])
        b=np.linspace(1,size[1],size[1])
        b=np.reshape(b,[-1,1])
        c=b.dot(a)
        dim=np.ones((size[0],1,size[-1]))
        mat=dim*c
        return tf.sin(mat)
        
    def mask(self,size):
        a=np.zeros(size)
        for i in range(len(a)):
            a[:,i,0:i+1]=1
        return tf.constant(a,dtype=tf.float32)
    
    def norm(self,x,name):
        with tf.variable_scope(name):
            mean,var=tf.nn.moments(x,[-1],keep_dims=True)
            standard_x=(x-mean)/tf.sqrt(var+1e-8)
            gamma=tf.get_variable(name='var',dtype=tf.float32,shape=[var.get_shape().as_list()[-1]],initializer=tf.random_normal_initializer(mean=1.,stddev=.02))
            beta=tf.get_variable(name='mean',dtype=tf.float32,shape=[mean.get_shape().as_list()[-1]],initializer=tf.constant_initializer(0.))
            out=gamma*standard_x+beta
            return out


def label_smooth(inputs,epsilon=0.1):
    K=inputs.get_shape().as_list()[-1]
    return ((1-epsilon)*inputs+(epsilon/k))

def train(batch_size,vocab_size,epoch):
    encoder_in=tf.placeholder(dtype=tf.float32,shape=[batch_size,None])
    decoder_in=tf.placeholder(dtype=tf.float32,shape=[batch_size,None])
    targets=tf.placeholder(dtype=tf.float32,shape=[batch_size,None])
    targets=label_smooth(targets)
    lr=tf.placeholder(dtype=tf.float32,shape=[])
    
    model=Transform(vocab_size=vocab_size)
    encoder_out=model.encoder(encoder_in)
    logits=model.decoder(decoder_in,encoder_out)
    loss=tf.contrib.seq2seq.sequence_loss(logits,targets,weights=tf.ones_like(targets))
    
    train_op=tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.98,epsilon=1e-09)
    op=train_op.minimize(loss)
    
    saver=tf.train.Saver(tf.global_variables())
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(epoch*):
            
            current_lr=min(step**(-0.5),step*4000**(-1.5))
            feed_dict={encoder_in:,decoder_in:,targets:,lr:curr_lr,model.attention_prob:,model.sublayer_prob:0.1,model.position_prob:}
            _,cost=sess.run([op,loss],feed_dict=feed_dict)
            if step%500==0:
                saver.save(sess,os.path.join(os.getcwd(),'attention\attention.ckpt'),global_step=i)
                print(cu)
        saver.save(sess,os.path.join(os.getcwd(),'attention\attention.ckpt'),global_step=i)
 
