from rest_framework.response import Response
from rest_framework.decorators import api_view
from . import recomendacao

@api_view(['GET'])
def getSimilar(request):

    img = request.GET.get('img',0)
    num = request.GET.get('num',4)

    recomend = recomendacao.busca_produtos(num,img)
    'imagens similares: {recomend} url processada: {str(img)}'
    retorno = {'imagens_similares': recomend, 'url_processada': str(img)}
    
    return Response(retorno)