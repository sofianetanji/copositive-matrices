default: nicetest

run:
	time (nice -n 19 python3 -u -W ignore::DeprecationWarning ./main.py) | tee ./main.output.txt ; notify-send "./main.py done"

nicetest:
	nice -n 19 python3 -u -W ignore::DeprecationWarning ./main.py | tee ./main.output.txt ; notify-send "./main.py done"

clean:
	-rm -rfv ./*.~ ./*.pyc ./__pycache__/

pylint:
	-pylint -d redefined-outer-name ./*.py | tee ./pylint_report.txt
	-pylint -f html -d redefined-outer-name ./*.py > ./pylint_report.html
	-html2markdown ./pylint_report.html > ./pylint_report.md

requirements:
	pip3 install -r requirements.txt