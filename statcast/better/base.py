import abc
from inspect import signature, Signature, Parameter

from sklearn.externals import joblib

from ..tools.fixpath import findFile

addParams = [Parameter('name', Parameter.POSITIONAL_OR_KEYWORD, default=''),
             Parameter('xLabels', Parameter.POSITIONAL_OR_KEYWORD, default=()),
             Parameter('yLabels', Parameter.POSITIONAL_OR_KEYWORD, default=())]
xMethods = ['predict', 'predict_proba', 'transform', 'decision_function',
            'predict_log_proba', 'score_samples']
xyMethods = ['fit', 'fit_predict', 'fit_transform', 'score', 'partial_fit']
fitMethods = ['fit', 'fit_predict', 'fit_transform', 'partial_fit']


def addXMethod(clsobj, method):
    '''Doc String'''

    if hasattr(clsobj, method) and not hasattr(clsobj, method + 'D'):
        methodSig = signature(getattr(clsobj, method))
        methodParams = [p for p in methodSig.parameters.values()]
        methodDParams = methodParams.copy()
        if methodDParams[0].name != 'self':
            methodDParams.insert(0,
                                 Parameter('self',
                                           Parameter.POSITIONAL_ONLY))
        if methodDParams[1].kind is Parameter.VAR_POSITIONAL:
            methodDParams.insert(1, Parameter('data',
                                              Parameter.
                                              POSITIONAL_OR_KEYWORD))
        else:
            methodDParams[1] = \
                Parameter('data', Parameter.POSITIONAL_OR_KEYWORD)

        methodDSig = Signature(methodDParams)

        def methodD(self, data, *args, **kwargs):
            X = self.createX(data)
            return getattr(self, method)(X, *args, **kwargs)
        methodD.__signature__ = methodDSig
        setattr(clsobj, method + 'D', methodD)


def addXYMethod(clsobj, method):
    '''Doc String'''

    if hasattr(clsobj, method) and not hasattr(clsobj, method + 'D'):
        methodSig = signature(getattr(clsobj, method))
        methodParams = [p for p in methodSig.parameters.values()]
        methodDParams = methodParams.copy()
        if methodDParams[0].name != 'self':
            methodDParams.insert(0,
                                 Parameter('self',
                                           Parameter.POSITIONAL_ONLY))
        if methodDParams[1].kind is Parameter.VAR_POSITIONAL:
            methodDParams.insert(1, Parameter('data',
                                              Parameter.
                                              POSITIONAL_OR_KEYWORD))
        else:
            methodDParams[1] = \
                Parameter('data', Parameter.POSITIONAL_OR_KEYWORD)
        if methodDParams[2] is Parameter.VAR_POSITIONAL:
            pass
        else:
            del methodDParams[2]
        methodDSig = Signature(methodDParams)

        def methodD(self, data, *args, **kwargs):
            X = self.createX(data)
            Y = self.createY(data)
            return getattr(self, method)(X, Y, *args, **kwargs)
        methodD.__signature__ = methodDSig
        setattr(clsobj, method + 'D', methodD)


def addXMethods(clsobj):
    '''Doc String'''

    for method in xMethods:
        addXMethod(clsobj, method)


def addXYMethods(clsobj):
    '''Doc String'''

    for method in xyMethods:
        addXYMethod(clsobj, method)


class BetterMetaClass(abc.ABCMeta):
    ''' Doc String'''

    def __new__(cls, clsname, bases, clsdict):
        '''Doc String'''

        clsobj = super().__new__(cls, clsname, bases, clsdict)
        oldInit = clsobj.__init__
        oldParams = [param for param in signature(oldInit).parameters.values()]

        firstIn = oldParams.pop(0)
        if firstIn.name != 'self':
            raise RuntimeError('BetterModels init must have self first '
                               'argument')
        if oldInit is object.__init__:
            oldParams = []
        if addParams == oldParams[:len(addParams)]:
            newParams = []
        else:
            newParams = addParams
        if any(nP == oP for nP in newParams for oP in oldParams):
            raise RuntimeError('BetterModels cannot define {} as inputs to '
                               'init'.format(', '.join([nP.name
                                                        for nP in newParams])))
        try:
            if any(p.kind != Parameter.POSITIONAL_OR_KEYWORD
                   for p in clsobj._params):
                raise RuntimeError('BetterModels can only add positional or '
                                   'keyword inputs to init from _params')
            elif any(p.default is Parameter.empty
                     for p in clsobj._params):
                raise RuntimeError('BetterModels init inputs must have '
                                   'defaults')
        except AttributeError:
            raise RuntimeError('BetterModels _params class attribute must be '
                               'a list of Parameters from the inspect module')
        newSig = Signature([firstIn] + newParams + clsobj._params + oldParams)

        def newInit(self, *args, **kwargs):
            '''Doc String'''

            bound = newSig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            bound.arguments.popitem(False)
            for dummy in range(len(newParams)):
                name, val = bound.arguments.popitem(False)
                setattr(self, name, val)
            for dummy in range(len(clsobj._params)):
                name, val = bound.arguments.popitem(False)
                setattr(self, name, val)

            oldInit(self, *bound.args, **bound.kwargs)

        newInit.__signature__ = newSig
        setattr(clsobj, '__init__', newInit)

        addXMethods(clsobj)
        addXYMethods(clsobj)

        return clsobj


class BetterModel(metaclass=BetterMetaClass):
    '''Doc String'''

    _params = []

    def createX(self, data):
        '''Doc String'''

        return data[self.xLabels]

    def createY(self, data):
        '''Doc String'''

        return data[self.yLabels]

    def save(self, path=None):
        '''Doc String'''

        if path is None:
            path = self.name
        joblib.dump(self, path + '.pkl')

    def load(self, name=None, filePath=None, searchDirs=None):
        '''Doc String'''

        if filePath is None:
            if name is None:
                name = self.name

            if searchDirs is None:
                filePath = findFile(name + '.pkl')
            else:
                filePath = findFile(name + '.pkl', searchDirs)

        return joblib.load(filePath)
