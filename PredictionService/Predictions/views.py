from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser
from .income_prediction import predict_income


# Create your views here.
class PredictionView(APIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (JSONParser,)

    def post(self, request):
        prediction = predict_income(request.data)
        content = {'prediction': prediction}
        return Response(content)
