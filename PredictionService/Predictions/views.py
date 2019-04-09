from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser
import importlib

module_name = input('data_handler')
importlib.import_module(module_name)


# Create your views here.
class PredictionView(APIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (JSONParser,)

    def post(self, request):
        # prediction = predict_income(request.data)
        content = {'prediction': request.data}
        return Response(content)
