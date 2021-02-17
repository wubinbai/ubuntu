class A(object):
    def __init__(self,a,b,c,d,e,f,g):
        #t = locals().items()
        #for k, v  in t:
        #    if k!='self':
        #        print('k',k)
        #        print('v',v)
        #        print('====')

        self.__dict__.update({k:v for k, v in locals().items() if k != 'self'})
