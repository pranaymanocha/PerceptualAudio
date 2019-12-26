function vol_test_continue(){
            document.getElementById('button1').disabled = true;
            window.location.replace("librispeech_middle.php?pretest=1&answer1="+answer1+"&answer2="+answer2+"&answer3="+answer3+"&curr_level="+curr_level+"&prog=0&press="+press+"&date_begin="+date_begin+"&entire_thing="+entire_thing+"&button_press_time="+button_press_time+"&assignmentId="+assignmentId+"&workerId="+workerId+"&hitId="+hitId+"&sigma1="+sigma1+"&sigma2="+sigma2+"&sigma3="+sigma3+"&question1="+question1+"&question2="+question2+"&question3="+question3); 
}


        function pretest_right_answer() {
            document.getElementById('button1').disabled = true;
            document.getElementById('button2').disabled = true;
            document.getElementById('button3').disabled = true;
            document.getElementById('button4').disabled = true;
            location.replace("librispeech_middle.php?warmup1=1&answer1="+answer1+"&answer2="+answer2+"&answer3="+answer3+"&curr_level="+curr_level+"&prog=0&press="+press+"&date_begin="+date_begin+"&entire_thing="+entire_thing+"&button_press_time="+button_press_time+"&assignmentId="+assignmentId+"&workerId="+workerId+"&hitId="+hitId+"&sigma1="+sigma1+"&sigma2="+sigma2+"&sigma3="+sigma3+"&question1="+question1+"&question2="+question2+"&question3="+question3);
        }

        function warmup_right_answer(){
            document.getElementById('button1').disabled = true;
            document.getElementById('button2').disabled = true;
            document.getElementById('playbtn').disabled = true;
            if (warmup2){
            window.location.replace("librispeech_middle.php?answer1="+answer1+"&answer2="+answer2+"&answer3="+answer3+"&curr_level="+curr_level+"&prog=0&press="+press+"&date_begin="+date_begin+"&entire_thing="+entire_thing+"&button_press_time="+button_press_time+"&assignmentId="+assignmentId+"&workerId="+workerId+"&hitId="+hitId+"&sigma1="+sigma1+"&sigma2="+sigma2+"&sigma3="+sigma3+"&question1="+question1+"&question2="+question2+"&question3="+question3);                   
            }
            else if (warmup1){
            window.location.replace("librispeech_middle.php?warmup2=1&answer1="+answer1+"&answer2="+answer2+"&answer3="+answer3+"&curr_level="+curr_level+"&prog=0&press="+press+"&date_begin="+date_begin+"&entire_thing="+entire_thing+"&button_press_time="+button_press_time+"&assignmentId="+assignmentId+"&workerId="+workerId+"&hitId="+hitId+"&sigma1="+sigma1+"&sigma2="+sigma2+"&sigma3="+sigma3+"&question1="+question1+"&question2="+question2+"&question3="+question3); 
            }
        }

        function warmup_wrong_answer() {
            document.getElementById('button1').disabled = true;
            document.getElementById('button2').disabled = true;
            alert("You choose the wrong answer. Please listen carefully to the audio files again to answer this question to proceed forward.");
        }

        function pretest_wrong_answer() {
            window.location.replace("fail.html");
        }

        function redirect(){
            //document.getElementById('pranay').innerHTML = prog;   
            window.location.replace("librispeech_middle.php?answer1="+answer1+"&answer2="+answer2+"&answer3="+answer3+"&curr_level="+curr_level+"&prog="+prog+"&press="+press+"&date_begin="+date_begin+"&entire_thing="+entire_thing+"&button_press_time="+button_press_time+"&assignmentId="+assignmentId+"&workerId="+workerId+"&hitId="+hitId+"&sigma1="+sigma1+"&sigma2="+sigma2+"&sigma3="+sigma3+"&question1="+question1+"&question2="+question2+"&question3="+question3);   
}
            

        function answers0() {
            //same
            document.getElementById('button1').disabled = true;
            document.getElementById('button2').disabled = true;
            document.getElementById('playbtn').disabled = true;
            button_press_time=button_press_time+ (new Date().getTime())/1000 +",";
            press = press+pressed+",";
            
            if (prog<=10){
                    question1 = question1+curr_level+",";
            
                    answer1 = answer1 + "0,";
                    if (prog==1){curr_level=1;}
                    else if (prog==10){curr_level=0;}
                    else if (prog >1 & prog<10){
                        [mu,sigma]=collect(question1,answer1,sigma1);
                        sigma1 = sigma1 + sigma+",";
                        var k=q_weights(question1,answer1);
                        var q=next_q(k,mu,sigma);
                        curr_level=q.toFixed(2);
                        
                    }
            }
            else if (prog>10 & prog<=20){
                    question2 = question2+curr_level+",";
                    answer2 = answer2 + "0,";
                    if (prog==11){curr_level=1;}
                    else if (prog==20){curr_level=0;}
                    else if (prog >11 & prog<20){
                         [mu,sigma]=collect(question2,answer2,sigma2);
                         sigma2 = sigma2 + sigma+",";
                         var k=q_weights(question2,answer2);
                         var q=next_q(k,mu,sigma);
                         curr_level=q.toFixed(2);
                    }    
            
            }
            else if (prog>20 & prog<=30){
                    question3 = question3+curr_level+",";
                    answer3 = answer3 + "0,";
                    if (prog==21){curr_level=1;}
                    else if (prog==30){curr_level=0;}
                    else if (prog >21 & prog<30){
                         [mu,sigma]=collect(question3,answer3,sigma3);
                         sigma3 = sigma3 + sigma+",";
                         var k=q_weights(question3,answer3);
                         var q=next_q(k,mu,sigma);
                         curr_level=q.toFixed(2);
                    }
            }
         
         redirect();
        }


        function answers1() {
          //different
            document.getElementById('button1').disabled = true;
            document.getElementById('button2').disabled = true;
            document.getElementById('playbtn').disabled = true;
            button_press_time=button_press_time+ (new Date().getTime())/1000 +",";
            press = press+pressed+",";
            
            if (prog<=10){
                    question1 = question1+curr_level+",";
                    answer1 = answer1 + "1,";
                    if (prog==1){curr_level=1;}
                    else if (prog==10){curr_level=0;}
                    else{
                        [mu,sigma]=collect(question1,answer1,sigma1);
                        sigma1 = sigma1 + sigma+",";
                        var k=q_weights(question1,answer1);
                        var q=next_q(k,mu,sigma);
                        curr_level=q.toFixed(2);
                       
                    }
            }
            else if (prog>10 & prog<=20){
                    question2 = question2+curr_level+",";
                    answer2 = answer2 + "1,";
                    if (prog==11){curr_level=1;}
                    else if (prog==20){curr_level=0;}
                    else {
                         [mu,sigma]=collect(question2,answer2,sigma2);
                         sigma2 = sigma2 + sigma+",";
                         var k=q_weights(question2,answer2);
                         var q=next_q(k,mu,sigma);
                         curr_level=q.toFixed(2);
                    }    
            
            }
            else if (prog>20 & prog<=30){
                    question3 = question3+curr_level+",";
                    answer3 = answer3 + "1,";
                    if (prog==21){curr_level=1;}
                    else if (prog==30){curr_level=0;}
                    else {
                         [mu,sigma]=collect(question3,answer3,sigma3);
                         sigma3 = sigma3 + sigma+",";
                         var k=q_weights(question3,answer3);
                         var q=next_q(k,mu,sigma);
                         curr_level=q.toFixed(2);
                    }
            }
         redirect();
        }

function q_weights(question,answer){
        var yes_count_w=0;
        var no_count_w=0;
        var que = question.split(",");
        var ans= answer.split(",");

            for(var i = (ans.length-1); i >=0; i--) {
                if (ans[i-1]==0){
                no_count_w+=Math.pow(1/2,ans.length-1-i);
                  //no_count
                }
              
              if (ans[i-1]==1){
                yes_count_w+=Math.pow(1/2,ans.length-1-i);
                  //yes_count
                }
              }
              k=yes_count_w-no_count_w;
              return k;
        }


        function next_q(k,mu,sigma){

              //k = yes.length - no.length;
              if (k>0){
              var power1=Math.pow(k,3)/2;
              var mean= mu - (power1*sigma)/2;
              mu1=mean;
                  
              }

              if (k<0){
              var power2=Math.pow(k,3)/2;
              var mean= mu - (power2*sigma)/2;
              mu1=mean;
                  
              }
              
              if (k==0){
              mu1=mu;
              }
              var randi_no=randn_bm();
              mu1=mu1+sigma*randi_no*0.30;

              if (mu1<0){
              mu1=0;}
              
              if (mu1>1){mu1=1;}
            

              return mu1;
        }

function randn_bm(){
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

        function collect(question,answer,sigma_record){
            var que = question.split(",");
            var ans= answer.split(",");
            var sigma_rec= sigma_record.split(",");

            var yes_count = 0; //count of yeses
            
            for(var k = 0;  k< (ans.length-1); k++){
                if(ans[k] == 1)
                    yes_count++;}
                                                   
            var no_count = ans.length-1-yes_count;

                var largest= -1e+06;
                            for (var i = 0.1; i < 0.9; i+=0.01) { 
                                for (var j = 0.01; j < 0.5; j+=0.01) { 
                                    var a=probMuSigma(i,j,que,ans,yes_count,no_count);
                                    if (largest<a){
                                            largest=a;
                                            var mu=i;
                                            var sigma=j;    
                              }
                        }
                    }
               
 
        var mu1=add_prior_m(mu,sigma,que,sigma_rec);
        var sig1=add_prior_s(mu,sigma,que,sigma_rec);
        
        return [mu1,sig1]; 
        }

        function probMuSigma(m,s,que,ans,yes_count,no_count){              
              
            

              var p = 0;
              var eps = 0.05;
              var beta = 0.50;
              
                p -= (beta * (Math.abs(cdf(0,m,s)-eps)));
                p -= (beta * (Math.abs(cdf(1,m,s)-(1-eps))));

                              for (var qa = 0; qa <Math.max((ans.length-3),0); qa++) {
                                
                                var ans_l = ans[ans.length-2-qa];
                                var que_l = que[que.length-2-qa];
                                    if (ans_l==1) {
                                    p += ((1-beta)*Math.log(cdf(que_l,m,s)))*Math.pow((1/(qa+1)),2);
                                }
                                else if (ans_l==0){
                                    p  += ((1-beta)*Math.log(1-cdf(que_l,m,s)))*Math.pow((1/(qa+1)),2);
                                }
                              }
                              return p;

        }


            function cdf(x,m,s) {
            if (x < m-(3*s)){ 
                return 0;
            }
            else if (x > m+(3*s)){ 
                return 1;
            }
            else {
                return 0.5*(1+erf((x-m)/(s*Math.sqrt(2))));
            }
        }

            function erf(x) {
            var m = 1;
            var s = 1;
            var sum = x * 1;
            for(var i = 1; i < 50; i++){
                m *= i;
                s *= -1;
                sum += (s * Math.pow(x, (2 * i) + 1)) / (m * ((2 * i) + 1));
                 }  
            return (2 * sum)/(Math.sqrt(Math.PI));
        }


        function normal1(x1){
          a = Math.exp((-x1*x1)/2)/(2*Math.PI);
          return a;
            }


        function pdf1(x, mu, sigma){ 
        a1 = normal1((x-mu)/sigma);
        return a1;
        }


        function add_prior_m(mu,sigma,mu_record,sigma_record){
         
       
        var mu_p=0.50;
        var sig_p=0.10;
            
        var sum=0;
        var a = mu_record.length;
        for (var i = 0; i < mu_record.length; i++){
            var ans = mu_record[i];
            sum=sum+ans*0.30;
        }
        var x_l = a;
        var lambda=Math.pow((x_l/10),2);
        if (a==0){a=1;}
        var mu11 = (1/((1/sig_p)+((1)/sigma)))*((mu_p/sig_p)+((mu)/sigma));
        if (lambda>=1){lambda=1;}
        mu = lambda*mu + (1-lambda)*mu11;
        return mu;
       
        }

        function add_prior_s(mu,sigma,mu_record,sigma_record){

        var x_l = sigma_record.length;
        var lambda= Math.pow((x_l/10),2);
        var sum=0;
        var a = x_l;
        for (var i = 0; i < sigma_record.length; i++) {
            var ans = sigma_record[i];
            sum=sum+ans*0.2;
        }
        if (a==0){
        a=1;
        }

        if (lambda>=1){lambda=1;}
        sigma = lambda*(sum/(a)) + (1-lambda)*(0.30);
            return sigma;
        }

var pressed=0;
$(document).ready(function(){
        
        var myAudio1 = document.getElementById('audio1');
        var myAudio2 = document.getElementById('audio2');

        function progressBar(event) {
          audio = event.target;
          
          var percent = Math.round(100.0 * audio.currentTime / audio.duration);
          var width = percent + '%';
          $(audio).next().css({ width: width });
        }

        function pauseThenStartSecondAudioClip() {
          if (voltest){
              $(button1).prop('disabled', false);
              setTimeout(function(){
            myAudio1.play();
          }, 600);
          }
          else if (pretest){
          document.getElementById('button1').disabled = false;
          document.getElementById('button2').disabled = false;
          document.getElementById('button3').disabled = false;
          document.getElementById('button4').disabled = false;
          }
          else {
          setTimeout(function(){
            myAudio2.play();
          }, 600);
        }
      }

        function showbuttons_restart() {
              $(button1).prop('disabled', false);
              $(button2).prop('disabled', false);
              $(playbtn).prop('disabled', false);
        }

        $(playbtn).click(function(){ 
            myAudio1.play();
            ++pressed;
            $(this).prop('disabled', true);
            $("#r2").css({ width: 0 });
        });

        $(myAudio1).bind("ended", pauseThenStartSecondAudioClip);
        $(myAudio2).bind("ended", showbuttons_restart);

        $("audio").bind("timeupdate", progressBar); 

});