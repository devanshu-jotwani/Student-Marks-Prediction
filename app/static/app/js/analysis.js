var error=document.getElementsByClassName("error");
var accuracy=document.getElementsByClassName("accuracy");

// console.log("error value",error[0].innerText)
// console.log("error type",typeof(error[0].innerText))
// console.log(accuracy)
var e=[]
for(let i =0;i<accuracy.length;i++)
{
    
    e[i]=parseFloat(error[i].innerHTML)
    // console.log(e[i])
    // console.log("Accuracy:",98.5-e[i])
    accuracy[i].innerHTML=98.5-e[i]   
}