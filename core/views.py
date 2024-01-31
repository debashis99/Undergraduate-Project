from django.shortcuts import render
from pipeline import img_to_latex_web

# Create your views here.


def index(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        latex = img_to_latex_web.img_to_latex(stream = img)
        print(latex)
        return render(request, 'response.html', {'latex' : latex})
    return render(request, "index.html", {})