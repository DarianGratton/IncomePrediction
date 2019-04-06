from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

# Create your views here.
class PredictionView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        prediction = 'placeholder'
        content = {'prediction': prediction}
        return Response(content)
