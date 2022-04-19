var pass=document.getElementById("dtr");
var lr=document.getElementById("lr");

var value=parseFloat(lr.innerHTML);
console.log(value)
if (value>6){
    pass.innerHTML="PASS"
} 