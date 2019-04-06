from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

# Create your views here.
class PredictionView(APIView):

    def post(self, request):
        prediction = 'placeholder'
        content = {'prediction': prediction}
        return Response(content)
